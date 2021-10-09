conda create -y -n sk python=3.8
conda activate sk
#source activate sktt
conda install scipy

####select according to your conda version####
####https://pytorch.org/####
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install pyaudio
conda install -c conda-forge librosa
pip install transformers==2.1.1

####If there is an error, try the command below with no version number
####
pip install fairseq==0.9.0
#pip install fairseq

pip install PyYAML
