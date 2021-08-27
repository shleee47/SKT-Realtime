# SKT-Realtime   

## Preparation    
### 1. Install the conda environment
```
cd SKT-Realtime/
conda install --file pyvad_list.txt
```    
### 2. Set the microphone index.
- If you run the code, you can check the microphone index. 
```   
cd SKT-Realtime/
python realtime_final.py
```          
- Set up the microphone index in the code 
```   
realtime_final.py line104
```       
![image](https://user-images.githubusercontent.com/57610448/130018053-5c8a48f4-50fa-4420-9a7d-3bacba97fc2e.png)

### 3. Fill out your ETRI access key.   
- Get the ETRI access key from the link below. It will take about a day to get approval.   
https://aiopen.etri.re.kr/key_main.php   
- Fill out the key in the code   
```
ETRI_STT.py line14
```
![image](https://user-images.githubusercontent.com/57610448/131095573-72b0de7e-bc65-48f7-ae53-a0abd7819694.png)


### 4. Run codes
```
cd SKT-Realtime/
python realtime_final.py
```     
