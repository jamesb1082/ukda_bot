from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, Input, Lambda
from keras.models import Sequential, Model
import os 
import sys 
import numpy as np 
from preprocess import get_data
from keras import backend as K 
from keras.optimizers import RMSprop 


def create_base_nn(embedding):
    seq = Sequential() 
    seq.add(embedding)
    seq.add(Dense(128,activation='relu'))
    seq.add(Dropout(0.1)) 
    seq.add(Dense(128,activation='relu'))
    seq.add(Dropout(0.1)) 
    seq.add(Dense(128,activation='relu')) 
    return seq    


def euclidean_distance(vects): 
    """
    Borrow from mnist siemese script. 
    """
    x,y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x- y), axis=1, keepdims=True), K.epsilon()))

def contrastive_loss(y_true, y_pred): 
    """
    Contrastive loss from mnist sieme script 
    """
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1-y_true) * K.square(
        K.maximum(margin - y_pred,0)))

def eucl_dist_output_shape(shapes): 
    shape1, shape2 = shapes
    return (shape1[0],1)


# variables 
glove_dir = '../../vectors/glove/' 
max_seq_len = 2500 
max_nb_words = 20000
embedding_dim = 100 
validation_split = 0.2 # what does valiation split mean? 


# pre-process the data
# The actual text pre-processing can be done in a seperate file  for the
# dataset that i will soon create.
print("Update: Preprocessing data") 
text_index, texts = get_data()

tokenizer = Tokenizer(num_words=max_nb_words,
        filters='#$%()*+,-./:;<=>?@[\\]^_{|}~\t\n', lower=True, split=" ") 

tokenizer.fit_on_texts(texts) 
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index 
data = pad_sequences(sequences, max_seq_len) 
print(data.shape)
labels = []

arr = np.zeros((len(text_index), 2, max_seq_len)) 

for li in text_index: 
    labels.append(li[2])
    # Put the data in the right shape
    q = sequences[li[0]] 
    a = sequences[li[1]]
    
for i in range(0,len(text_index)): 
    row = text_index[i]
    print(data[row[0]].shape)
    arr[i][0] = data[row[0]]
    arr[i][1] = data[row[1]] 


print(arr.shape)
# Indexing word vectors 
print("Update: Indexing word vectors") 
embed_index = {} 
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt')) 
for line in f: 
    values = line.split() 
    word = values[0] 
    coefs = np.asarray(values[1:], dtype='float32') 
    embed_index[word]=coefs
f.close()

# Generating vector embeddings
num_words = min(max_nb_words, len(word_index)+1)
embedding_matrix = np.zeros((num_words, embedding_dim)) 
print("Update: Generating vector embeddings")
for word, i in word_index.items(): 
    if i >=max_nb_words:
        continue 
    embedding_vector=embed_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector


question_input = Input(shape=(max_seq_len,))
answer_input = Input(shape=(max_seq_len,))

# Create embedding layer

print("Update: Creating Neural Network") 
embedding_layer = Embedding(num_words, 
        embedding_dim,
        weights=[embedding_matrix], 
        input_length=max_seq_len,
        trainable=False)

base_nn = create_base_nn(embedding_layer)

print(type(base_nn)) 

# Using the same input for both? As it seems to be just about dimensions
q_nn = base_nn(question_input)  
a_nn = base_nn(answer_input) 

distance = Lambda(euclidean_distance,
        output_shape=eucl_dist_output_shape)([q_nn,a_nn])

model = Model([question_input, answer_input], distance)

# compile and fit
rms = RMSprop() 
model.compile(loss=contrastive_loss, optimizer=rms) 

model.fit([arr[:,0], arr[:,1]], labels, batch_size=128, epochs=5) 



