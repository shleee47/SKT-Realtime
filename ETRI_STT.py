#-*- coding:utf-8 -*-
import urllib3
import json
import base64
import time
import pdb
import numpy as np

class STT_model:
    def __init__(self, config_path):
        super().__init__()
        self.openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
        '''Fill out your ETRI access key here'''
        self.accessKey = ""
        self.languageCode = "korean"

    def inference(self, counter, audio):
        self.audioContents = base64.b64encode(audio).decode("utf8")
 
        requestJson = {
            "access_key": self.accessKey,
            "argument": {
                "language_code": self.languageCode,
                "audio": self.audioContents
            }
        }
 
        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            self.openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )
 
        #print("[responseCode] " + str(response.status))
        #print("[responBody]")
        data = json.loads(response.data.decode("utf-8", errors='ignore'))    
        try:
            print(data['return_object']['recognized'])   
            return data['return_object']['recognized']
        except: 
            return None
