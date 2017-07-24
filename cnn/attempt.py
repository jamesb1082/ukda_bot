from __future__ import print_function 
import os 
import sys
import numpy as np 
from keras.preprocessing.text import text_to_word_sequence as ttws, Tokenizer
from keras.layers import Embedding, Dense, Dropout
from keras.models import Sequential 

def create_base_network(dimension, embed): 
    seq = Sequential()
    seq.add(embed) 
    #seq.add(Dense(128,input_shape=(dimension,), activation='relu')) 
    seq.add(Dropout(0.1)) 
    seq.add(Dense(128,activation='relu')) 
    seq.add(Dropout(0.1)) 
    seq.add(Dense(128,activation='relu')) 
    return seq 


#==============================================================================
# Main Program 
#============================================================================== 

# Variables 
BASE_DIR = '../../vectors' 
GLOVE_DIR = BASE_DIR + '/glove/' 
MAX_SEQUENCE_LENGTH = 1000 
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100 
VALIDATION_SPLIT = 0.2 

# Get the glove embeddings 
# first entry of glove vector is actually the word itself. 
# This acts as a dictionary key. 
embed_index = {} 
print("Indexing glove vectors.") 
f = open(os.path.join(GLOVE_DIR,'glove.6B.100d.txt')) 
for line in f: 
    values = line.split() 
    word = values[0] 
    coefs = np.asarray(values[1:], dtype='float32') 
    embed_index[word] = coefs 
f.close()

# Sample Texts and pre-process  
print("Pre-processing text.") 

texts = ["the cat sat on the mat, ",  
        "A difference in opinion is important,", 
        "What is the difference between an animal and a door?"] 
words = []

tokenizer = Tokenizer(num_words = MAX_NB_WORDS) 
tokenizer.fit_on_texts(texts) 
word_index = tokenizer.word_index 

labels = [0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1]



#convert words to list of words. 
for text in texts: 
    words += ttws(text, filters='!,.?', lower=True, split=" ") 


#  build an embedding matrix of width 100 by height num_words 
print("Create embedding matrix")

embed_matrix = [] 

num_words = min( MAX_NB_WORDS, len(words)) # gets whatever is smaller. 

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) 

for word, i in word_index.items(): 
    if i>=MAX_NB_WORDS: 
        continue 
    embedding_vector = embed_index.get(word) 
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector 



# NEURAL NETWORK STUFF 

# creating embedding layer - note trainable is set to false 

embedding_layer = Embedding(num_words, 
        EMBEDDING_DIM, 
        weights=[embedding_matrix],
        input_length = MAX_SEQUENCE_LENGTH, 
        trainable=False) 

# using the same network for both parts 
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') 

embedded_sequences = embedding_layer(sequence_input) 



