import requests
import json
import base64

class STT_model:
    def __init__(self, config_path):
        super().__init__()
        self.kakao_speech_url = "https://kakaoi-newtone-openapi.kakao.com/v1/recognize"
        
        '''fill out your api key here'''
        rest_api_key = ""

        self.headers = {
            "Content-Type": "application/octet-stream",
            "X-DSS-Service": "DICTATION",
            "Authorization": "KakaoAK " + rest_api_key,
        }

    #def inference(self, counter, audio_in_bytes):
    def inference(self, counter, audio_path):
        with open(audio_path, 'rb') as fp:
            audio = fp.read()
        #print(audio)
        #print(type(audio_in_bytes))
        #audio = base64.b64encode(audio_in_bytes)
        res = requests.post(self.kakao_speech_url, headers=self.headers, data=audio)
        result_json_string = res.text[res.text.index('{"type":"finalResult"'):res.text.rindex('}')+1]
        result = json.loads(result_json_string)
        #print(result)
        print(result['value'])
        return result['value']
