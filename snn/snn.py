from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, Input, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation
from keras.models import Sequential, Model
import os 
import sys 
import numpy as np 
from preprocess import get_data
from keras import backend as K 
from keras.optimizers import RMSprop 
from keras.utils import plot_model
from sklearn.metrics import classification_report 
import seaborn as sns 
import pandas 
def create_base_nn(embedding):
    """
    Adapted from the mnist siamese script. Returns a sequential model, with the
    first layer being a frozen embedding layer. 

    Return:
    -------
    A sequential model. 
    """
    seq = Sequential() 
    seq.add(embedding)
    seq.add(Dense(128,activation='relu'))
    seq.add(Dropout(0.1)) 
    seq.add(Dense(128,activation='relu'))
    seq.add(Dropout(0.1)) 
    seq.add(Dense(128,activation='relu')) 
    return seq    


def create_base_nn_updated(embedding): 
    """
    Same as the above function, however it is deeper.  
    """
    filters = 250 
    kernel_size = 5

    seq = Sequential() 
    seq.add(embedding)
    seq.add(Dropout(0.1))
    seq.add(Conv1D(filters, kernel_size,padding='valid',activation='relu', strides=1))
    seq.add(GlobalMaxPooling1D())
    seq.add(Dense(256))
    seq.add(Dropout(0.1)) 
    seq.add(Activation('relu')) 
    seq.add(Dropout(0.1)) 
    seq.add(Dense(1))
    seq.add(Activation('sigmoid'))
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
    return K.mean((1-y_true) * K.square(y_pred) + y_true * K.square(
        K.maximum(margin - y_pred,0)))



def eucl_dist_output_shape(shapes): 
    shape1, shape2 = shapes
    return (shape1[0],1)

def load_data():
    """
    This function loads the dataset and preprocesses so it is in the correct
    format. 

    Returns
    -------
    train_data: 3D numpy array. 
    test_data: 3D numpy array. 
    train_labels: 1D numpy array. 
    test_labels: 1D numpy array. 
    word_index: dictionary for words to indexes. 

    """
    text_index, texts = get_data()
    max_nb_words = 200000

    tokenizer = Tokenizer(num_words=max_nb_words,
            filters='#$%()*+,-./:;<=>?@[\\]^_{|}~\t\n', lower=True, split=" ") 

    tokenizer.fit_on_texts(texts) 
    sequences = tokenizer.texts_to_sequences(texts)
    value =0
    for i in sequences: 
        if len(i) > value: 
            value = len(i) 
    print(value) 
    word_index = tokenizer.word_index 
    data = pad_sequences(sequences, 2300) 
    labels = []
    # build a 3d numpy array to fit input type 
    arr = np.zeros((len(text_index), 2, max_seq_len)) 
    for i in range(0,len(text_index)): 
        row = text_index[i]
        arr[i][0] = data[row[0]]
        arr[i][1] = data[row[1]] 
        temp = text_index[i] 
        labels.append(temp[2]) 

    #split data into training and test sets. 
    labels = np.array(labels, dtype='int32') 
    train_val = int(len(arr) * (1-validation_split)) 
    train_data = arr[:train_val,:]
    test_data = arr[train_val:, : ] 
    train_labels = labels[:train_val]
    test_labels = labels[train_val:] 
    print(train_data[:,0].shape) 
    print(test_data.shape) 
    temp  =[test_data[:,0], test_data[:,0]]

    for i in train_data[:,0]:
        print(i) 
    return train_data,test_data,train_labels,test_labels, word_index 

def index_vectors(glove_dir): 
    embed_index = {} 
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt')) 
    for line in f: 
        values = line.split() 
        word = values[0] 
        coefs = np.asarray(values[1:], dtype='float32') 
        embed_index[word]=coefs
    f.close()
    return embed_index

def compute_acc(pred, labels): 
    """
    Borrowed from mnist siamese script! 
    """
    return labels[pred.ravel() < 0.5].mean() 

if __name__ == '__main__':  
    # variables 
    glove_dir = '../../vectors/glove/' 
    max_seq_len = 2300 
    max_nb_words = 200000
    embedding_dim = 100 
    validation_split = 0.2 # what does valiation split mean? 

    print("Update: Preprocessing data") 
    train_data, test_data, train_labels, test_labels, word_index= load_data() 

    print("Update: Indexing word vectors") 
    embed_index = index_vectors(glove_dir)  

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

    # Creating the inputs
    question_input = Input(shape=(max_seq_len,))
    answer_input = Input(shape=(max_seq_len,))

    # Create embedding layer
    print("Update: Creating Neural Network") 
    embedding_layer = Embedding(num_words, 
            embedding_dim,
            weights=[embedding_matrix], 
            input_length=max_seq_len,
            trainable=True)

    base_nn = create_base_nn_updated(embedding_layer)
    # Using the same input for both? As it seems to be just about dimensions
    q_nn = base_nn(question_input)  
    a_nn = base_nn(answer_input) 

    distance = Lambda(euclidean_distance,
            output_shape=eucl_dist_output_shape)([q_nn,a_nn])

    model = Model([question_input, answer_input], distance)

    # compile and fit
    rms = RMSprop() 
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy']) 
    history = model.fit([train_data[:,0], train_data[:,1]], train_labels, 
            batch_size=32, epochs=1000, validation_split=0.2) 

    # Predict and evaluate.
    pred = model.predict([train_data[:,0], train_data[:,0]]) 
    print("Prediction shape: ", pred.shape)
    train_out = model.evaluate([train_data[:,0], train_data[:,1]] , train_labels, batch_size=32) 
    pred2 = model.predict([test_data[:,0], test_data[:,0]]) 
    test_out = model.evaluate([test_data[:,0], test_data[:,1]] , test_labels, batch_size=32) 
    
    print() 

    print("=======================================================================") 
    
    # Check to see if all outputs are the same or not 
    tmp_vals = [] 
    
    for value in pred:
        print(value) 
        tmp_vals.append(float(value[0])) 
    print("Are all outputs the same: ", len(set(tmp_vals))== 1) 
    print("=======================================================================") 
    print("Training classification report")  
    print(classification_report(train_labels, map(round,pred))) 
    print() 

    print("Testing classification report") 
    print(classification_report(test_labels, map(round,pred2))) 
    print("=======Results===========") 
    print("Training set tests:")
    for i in range(len(train_out)): 
        print(model.metrics_names[i], ': ', round(train_out[i],5)) 
    print("Test set tests:") 
    for i in range(len(test_out)): 
        print(model.metrics_names[i], ': ', round(test_out[i],5))
    
    # plot some graphs 
    dt  = history.history['acc'] 
    data = pandas.DataFrame({"acc":dt})  
    ax = sns.tsplot(data=data["acc"] )
    ax.set(xlabel="epoch", ylabel="Accuracy") 
    sns.plt.show() 
