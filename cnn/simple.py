from keras.preprocessing.text import Tokenizer, text_to_word_sequence as ttws
from keras.layers import Dense 
from keras.models import Sequential
import os 
import numpy as np 
import sys


def generate_labels(t_letter, sequences): 
    labels = [] #lists of labels 
    
    for word in sequences: 
        if t_letter in word:
            labels.append(1) 
        else:
            labels.append(0) 
    return labels



# Step1: Preprocess the data and generate relevant labels. 
max_words = 1000


texts = ["this is a sentence.", "wow what a great dog", 
        "what is machine learning", "how do you build a neural network", 
        "please tell someone to help me", "I am awfully stuck at the moment, how about you"
        ]
tokenizer = Tokenizer(num_words = max_words, filters='!,.?', lower=True, split=" ")
tokenizer.fit_on_texts(texts) 
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts) 

words = [] 
for text in texts: 
    words+= ttws(text, filters='.,!?', lower=True, split=" ") 

labels = generate_labels('a', words)   


# Step 2: Generate vector embeddings 
glove_dir = '../../vectors/glove/' 
embedding_dim = 100 

embed_index = {} 
f = open(os.path.join(glove_dir,'glove.6B.100d.txt') ) 
for line in f: 
    values = line.split() 
    word = values[0] 
    coefs= np.asarray(values[1:], dtype='float32') 
    embed_index[word]=coefs
f.close() 
 

# Step 2.2 Convert words into vectors. 
embed_vectors = np.zeros((len(words), embedding_dim)) 

for word in words: 
    vector = embed_index[word] 
    embed_vectors[word_index[word]] = vector 

print(embed_vectors.shape) 

# Step 3: Create a neural network and train neural network. 
model = Sequential() 
model.add(Dense(32, input_dim=100, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1,activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(embed_vectors,labels, epochs=150,batch_size=10) 


# Step 4: Predict 
