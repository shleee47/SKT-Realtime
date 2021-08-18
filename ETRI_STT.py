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
        #self.accessKey = ""
        #self.accessKey = "fe30fafc-5d75-47cd-9f87-8c520977385d"
        #self.accessKey = "03727d41-f227-4b39-9a40-1035acd2d5c7"
        self.accessKey = "0368411d-3ff7-4ae7-8eaf-94f84a1bde98"
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
