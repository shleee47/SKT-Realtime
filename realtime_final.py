import numpy as np
import pyaudio
import scipy.io.wavfile as wav
import threading
from multiprocessing import Process, Queue, Array
import pdb
import warnings
import argparse
import time
import os, sys
import subprocess
import pickle
import time
from demo import emotion_model
from demo_text import emotion_text_model
from demo_audio import emotion_audio_model
from ETRI_STT import STT_model
#from kakao import STT_model
import librosa
#import webrtcvad
#import soundfile

def start_stream(Queue_audio):
    while True:
        data = stream.read(CHUNK_SIZE)
        Queue_audio.put(np.fromstring(data, dtype=np.int16))

def start_process(Queue_audio):
    counter = 0
    text = None
    flag = False
    
    while True:
        new_frame = Queue_audio.get()

        vad = np.abs(np.asarray(new_frame[4::16])).mean() > 30
        if vad == True:
            if flag == False:
                flag = True
                print("========Started!!!=========")
                channel0 = np.asarray(new_frame[4::16])
            else:
                channel0 = np.concatenate((channel0,np.asarray(new_frame[4::16])),axis=0)
            
            audio_in_bytes = channel0.tobytes()
            text = STT.inference(counter,audio_in_bytes)
            if text != None:
                multimodal = threading.Thread(target=multi_model, args=(counter,channel0,text))
                onlytext = threading.Thread(target=text_model, args=(counter,channel0,text))
                onlyaudio = threading.Thread(target=audio_model, args=(counter,channel0,text))
            
                multimodal.start()
                onlytext.start()
                onlyaudio.start()

                multimodal.join()
                onlytext.join()
                onlyaudio.join()
                #with open(os.path.join('/home/nas3/user/sanghoon/code/SKT-realtime/output/end.txt'),'w',encoding='utf8') as f:
                #    f.write('ready for GUI output\n')
                #f.close()
                #EMOTION.inference(counter,[channel0.astype(np.float32)], [text])
        else:
            print("========Finished!!!=========")
            flag = False
            text = None
        counter+=1
        
def multi_model(counter,channel0,text):
    EMOTION.inference(counter,[channel0.astype(np.float32)], [text])

def text_model(counter,channel0,text):
    EMOTION_TEXT.inference(counter,[channel0.astype(np.float32)], [text])

def audio_model(counter,channel0,text):
    EMOTION_AUDIO.inference(counter,[channel0.astype(np.float32)], [text])

if __name__ == '__main__':
    #warnings.filterwarnings("ignore")
    config_path = './config/config.yml'
    config2_path = './config/config2.yml'
    config3_path = './config/config3.yml'
    RATE = 16000
    CHUNK_SIZE = 16000

    '''Network Initialization'''
    STT = STT_model(config_path)
    EMOTION = emotion_model(config_path)
    EMOTION_TEXT = emotion_text_model(config2_path)
    EMOTION_AUDIO = emotion_audio_model(config3_path)
    
    # Audio Setting
    Queue_audio = Queue()

    pypy = pyaudio.PyAudio()
    print('============================================')
    print(pypy.get_device_count())
    print('============================================')
    for index in range(pypy.get_device_count()):
        desc = pypy.get_device_info_by_index(index)
        print("DEVICE: %s INDEX: %s RATE:%s"%(desc['name'],index,int(desc["defaultSampleRate"])))

    #pdb.set_trace()
    stream = pypy.open(format=pyaudio.paInt16, channels=16, rate=RATE, input=True, input_device_index=3,frames_per_buffer=CHUNK_SIZE)
    #pdb.set_trace()

    '''Initial Parameter Setting'''
    counter = 0
    
    while True:
        
        streaming = threading.Thread(target=start_stream, args=(Queue_audio,))
        processing = threading.Thread(target=start_process, args=(Queue_audio,))

        streaming.start()
        processing.start()

        streaming.join()
        processing.join()
