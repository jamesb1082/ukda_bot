from keras.preprocessing.text import Tokenizer, text_to_word_sequence as ttws
from keras.layers import Dense 
from keras.models import Sequential
import os 
import numpy as np 
import sys
from random import shuffle 
from keras.utils import plot_model 

def generate_labels(t_letter, sequences): 
    labels = [] #lists of labels 
    
    for word in sequences: 
        if t_letter in word:
            labels.append(1) 
        else:
            labels.append(0) 
    return labels



# Step1: Preprocess the data and generate relevant labels. 
max_words = 100000
print("Preprocessing the data") 

texts = [] 
f = open('data.txt', 'r') 
for line in f: 
    texts.append(line.strip('\n'))  
f.close()
for i in range(5): 
    shuffle(texts) 

tokenizer = Tokenizer(num_words = max_words, filters='!,.?', lower=True, split=" ")
tokenizer.fit_on_texts(texts) 
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts) 

words = [] 
for text in texts: 
    words+= ttws(text, filters='.,!?', lower=True, split=" ") 

labels = generate_labels('a', words)   
training_labels = labels[:int(len(labels)*0.8)] 
test_labels = labels[int(len(labels)*0.8):] 


# Step 2: Generate vector embeddings
print("Generating Vector embeddings") 
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
    try:
        vector = embed_index[word] 
        embed_vectors[word_index[word]] = vector 
    except: 
        continue
training_data = embed_vectors[:int(len(embed_vectors)* 0.8) ] 
test_data = embed_vectors[int(len(words)*0.8):] 

# Step 3: Create a neural network and train neural network. 
model = Sequential() 
model.add(Dense(32, input_dim=100, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(1,activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(training_data, training_labels, epochs=10, batch_size=16)  

# Step 4: Predict
#pred = model.predict(test_data,
print(len(test_data)) 
print(len(test_labels))
score = model.evaluate(test_data, test_labels) 
print(score)

plot_model(model, to_file="model.png") 
