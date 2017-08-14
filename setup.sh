#!/bin/bash 
#Set up python virtual env first. 

# =============================================================================
# Python Packages 
# =============================================================================
pip install numpy 
pip install scipy
pip install pandas
pip install sklearn
pip install chatterbot
pip install matplotlib
pip install seaborn
pip install gensim 
pip install jupyter
pip install tensorflow-gpu
pip install keras
pip install h5py
pip install pickle
pip install progressbar
pip install flask 
# =============================================================================
# download glove vectors to correct location  
# =============================================================================
mkdir vectors
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip 
rm glove.6B.50d.txt
rm glove.6B.200d.txt
rm glove.6B.300d.txt 
rm glove.6B.zip
mkdir vectors/glove
mv glove.6B.100d.txt vectors/glove/
# =============================================================================
# download word2vec vectors to correct location  
# =============================================================================
./misc/download_word2vec.sh
mkdir vectros/word2vec
gzip -d GoogleNews-vectors-negative300.bin.gz
rm GoogleNews-vectors-negative300.bin.gz 
mv GoogleNews-vectors-negative300.bin vectors/word2vec

python test.py
