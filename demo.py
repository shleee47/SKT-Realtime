import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
import torch
import torch.nn as nn
import re
import pdb
import librosa
import html
from KoBERT.tokenization import BertTokenizer
from models import load_bert
import yaml 
from torch.utils.data.dataset import Dataset
import numpy as np
import time
from torch.utils.data import DataLoader
import os
import pickle
from pathlib import Path
from models import MultimodalTransformer

'''Tester''' 
class ModelTester:
    def __init__(self, model, config):

        self.device = torch.device('cuda:{}'.format(config['demo']['device']))
        self.model = model.cuda()
        self.model.eval()
        self.n_class = config["transformer"]["n_classes"]
        self.class_list = ['negative','neutral','positive']
        self.load_checkpoint(config['demo']['ckpt_path'])
        
        ## Text Feature Extractor
        self.bert = load_bert('./KoBERT', self.device)
        self.bert.eval()
        self.bert.zero_grad()
        self.save_file_path = './output2/'

    def load_checkpoint(self, ckpt):
        print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def test(self, test_loader,text,counter):
        #print('test starts')
        self.test_loader = test_loader
        #batch_size = len(self.test_loader) 
        batch_size = 1
        
        with torch.no_grad():
            for b, batch in enumerate(self.test_loader):

                batch = map(lambda x: x.cuda() if x is not None and type(x[0]) != str else x, batch)
                audios, a_mask, texts, t_mask = batch
                texts, _ = self.bert(texts, ~t_mask)
                outputs, _ = self.model(audios, texts, a_mask, t_mask)
                best_prediction = outputs.max(-1)[1]

                print('multimodal: {}'.format(self.class_list[best_prediction.item()]))
                #with open(os.path.join(self.save_file_path,'multimodal.txt'), 'w',encoding='utf8') as f:
                #    f.write(text[0]+'\n'+self.class_list[best_prediction.item()]+'\n')
                #f.close()

'''Dataloader'''
class Data_Reader(Dataset):
    def __init__(self, tokenizer, vocab, only_audio, only_text):
        super(Data_Reader, self).__init__()
        self.class_list = ['negative','neutral','positive']
        self.nfft = 1024
        self.n_mfcc = 40
        self.hopsize = self.nfft // 4
        self.window = 'hann'
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.only_text = only_text
        self.only_audio = only_audio
        #self.pase = 

        ## Special Tokens
        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def audio_and_text(self,audio_list,text_list):
        self.audio_list = audio_list
        self.text_list = text_list

    def __len__(self):
        return len(self.audio_list)

    @staticmethod
    def normalize_string(s):
        s = html.unescape(s)
        s = re.sub(r"[\s]", r" ", s)
        s = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", s)
        return s

    def MFCC(self, sig): 
        def mfcc(sig):   
            S = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=self.n_mfcc,
                    n_fft = self.nfft, hop_length = self.hopsize)
            return S

        return mfcc(sig)

    def __getitem__(self, idx):
        #pdb.set_trace()
        #audio_path = self.audio_list[idx]
        audio = self.audio_list[idx]
        text = self.text_list[idx]
        feature, token_ids = None,None

        '''Feature Extraction'''
        if not self.only_audio:
            tokens = self.normalize_string(text)
            tokens = self.tokenizer.tokenize(tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        if not self.only_text:
            #check5 = time.time()
            #audio, _ = librosa.load(audio_path, sr=16000, dtype=np.float32)
            #print("librosa load time: {}".format(time.time()-check5))
            #feature = self.pase(audio)  1 x 256 x T  
            audio /= max(abs(audio))    
            feature = self.MFCC(audio)
        else: 
            return None, token_ids#, audio_path

        return torch.FloatTensor(feature).transpose(0,1), token_ids#, audio_path

'''Collate Function'''
class Data_Collate:
    def __init__(self,
                 pad_idx,
                 cls_idx,
                 sep_idx,
                 bert_args_path,
                 config,
                 device='cpu'):
        self.device = device #cpu
        self.n_mfcc = 40
        self.n_classes = config['n_classes']

        self.max_len_bert = torch.load(bert_args_path).max_len
        self.pad_idx = pad_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.only_text = config['only_text']
        self.only_audio = config['only_audio']

    def __call__(self, batch):
        #audios, sentences, audio_path = list(zip(*batch))
        audios, sentences = list(zip(*batch))
        audio_emb, audio_mask, text_emb, text_mask = None, None, None, None
        with torch.no_grad():

            if not self.only_text:
                audio_emb, audio_mask = self.audio_pad_mask(audios)

            if not self.only_audio:
                input_ids = torch.tensor([self.text_pad_mask(sent, self.max_len_bert) for sent in sentences])
                text_mask = torch.ones_like(input_ids).masked_fill(input_ids == self.pad_idx, 0).bool()
            else: 
                return audio_emb, audio_mask, None, None#, audio_path
        
        return audio_emb, audio_mask, input_ids, ~text_mask#, audio_path

    def audio_pad_mask(self,audios):

        audio_len = [len(x) for x in audios]
        B = len(audios)
        T = max(audio_len) 
        F = audios[0].size(1)
        padded = torch.zeros(B, T, F).fill_(float('-inf'))
        for i in range(B):
            padded[i, :audios[i].size(0),:] = audios[i]

        # get key mask
        mask = padded[:, :, 0]
        mask = mask.masked_fill(mask != float('-inf'), 0)
        mask = mask.masked_fill(mask == float('-inf'), 1).bool()
        
        # -inf -> 0.0
        padded = padded.masked_fill(padded == float('-inf'), 0.)
        return padded, mask#, arr_label
    

    def text_pad_mask(self, sentence, max_len):
        sentence = [self.cls_idx] + sentence + [self.sep_idx]
        diff = max_len - len(sentence)
        if diff > 0:
            sentence += [self.pad_idx] * diff
        else:
            sentence = sentence[:max_len - 1] + [self.sep_idx]
        return sentence


'''Demo'''
class emotion_model:

    def __init__(self, config_path):
        super().__init__()
        #pdb.set_trace()
        os.environ["CUDA_VISIBLE_DEVICES"]='0'
        
        ## Load Config
        with open(os.path.join(config_path), mode='r') as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)
        f.close()

        ## Load Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert']['vocab_path'], do_lower_case=False)
        self.vocab = self.tokenizer.vocab

        ## Model Initialization
        self.model = MultimodalTransformer(**self.config['transformer'])

        ## Tester Initialization
        self.tester = ModelTester(self.model, self.config)

        ## Dataloader
        self.dataset = Data_Reader(self.tokenizer, self.vocab, self.config['transformer']['only_audio'],self.config['transformer']['only_text'])
        fn_collate = Data_Collate(self.vocab['[PAD]'], self.vocab['[CLS]'], self.vocab['[SEP]'], self.config['bert']['args_path'], self.config['transformer'])
        self.loader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=False, collate_fn=lambda x: fn_collate(x), pin_memory = True, num_workers=0)


    def inference(self, counter, audio, text):
        ## Fill in New Audio and Text to dataloader
        self.dataset.audio_and_text(audio, text)
        
        ## Inference
        self.tester.test(self.loader,text,counter)

