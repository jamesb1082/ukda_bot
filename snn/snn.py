from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, Input, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, AveragePooling1D, MaxPooling1D
from keras.optimizers import Adam  
import os
import numpy as np
from preprocess import get_data, get_file_links, get_raw_strings
from keras import backend as K
import seaborn as sns 
import pandas
import argparse
from evaluate import evaluation


def create_base_nn_updated(embedding): 
    """
    Same as the above function, however it is deeper.  
    """
    filters = 256 
    kernel_size = 3
    d_value = 0.13 
    seq = Sequential() 
    seq.add(embedding)
    seq.add(Dropout(d_value))
    seq.add(Conv1D(filters, kernel_size,padding='valid',activation='relu', strides=1))
    #seq.add(Flatten())  
    seq.add(GlobalMaxPooling1D()) 
    for i in range(1):
        seq.add(Dense(128))
        seq.add(Dropout(d_value))  
        seq.add(Activation('elu'))
        seq.add(Dropout(d_value))
    
    seq.add(Dense(1))
    seq.add(Activation('relu')) 
    return seq    

def euclidean_distance(vects): 
    """
    Borrow from mnist siamese script. 
    """
    x,y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x- y), axis=1, keepdims=True), K.epsilon()))


def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from mnist siamese script however it has been modified. 
    
    Found a github issue stating that (1-y_true) and y_true should be swapped round. 
    originally they were the the other way. 

    """
    margin = 1
    return K.mean((1-y_true) * K.square(y_pred) + (y_true) * K.square(
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
    temp  =[test_data[:,0], test_data[:,0]]    
    return train_data,test_data,train_labels,test_labels, word_index, data 


def index_vectors(glove_dir):
    """
    indexes the glvoe vectors as a dictionary
    """
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


def create_embedding(word_index, glove_dir, max_nb_words):
    """
    Generates the embedding layer and returns it 
    """
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

       # Create embedding layer
    print("Update: Creating Neural Network") 
    embedding_layer = Embedding(num_words, 
            embedding_dim,
            weights=[embedding_matrix], 
            input_length=max_seq_len,
            trainable=False)
    return embedding_layer


def display_results(train_out, test_out, model):
    """
    Displays the training and testing results from model.evaluate
    """
    print()  
    # Check to see if all outputs are the same or not  
    print() 
    print("=======Results===========") 
    print("Training set tests:")
    for i in range(len(train_out)): 
        print(model.metrics_names[i], ': ', round(train_out[i],5)) 
    print("Test set tests:") 
    for i in range(len(test_out)): 
        print(model.metrics_names[i], ': ', round(test_out[i],5))


def training_graph(history): 
    """
    Displays a graph showing acc vs epochs
    """
    dt  = history.history['acc'] 
    data = pandas.DataFrame({"acc":dt})      
    ax = sns.tsplot(data=data["acc"] )
    ax.set(xlabel="epoch", ylabel="Accuracy") 
    sns.plt.show() 


if __name__ == '__main__':  
    parser = argparse.ArgumentParser("Siamese neural network for question answering")
    parser.add_argument("-l", "--load", help="Load a neural network", 
            action="store", type=str) 
    args = parser.parse_args() 
    
    
    # ==========================================================================
    # variables 
    # ==========================================================================
    glove_dir = '../../vectors/glove/' 
    max_seq_len = 2300 
    max_nb_words = 200000
    embedding_dim = 100 
    validation_split = 0.2 
    save_file = 'saved_models/snn.h5'
    epochs = 100
    bs = 128 #batch size 
    
    # ==========================================================================
    # Pre-process the data
    # ==========================================================================
    print("Update: Preprocessing data") 
    train_data, test_data, train_labels, test_labels, word_index, sequences= load_data() 
    # TRAINING ON ALL THE DATA. ATTEMPTING TO OVERFIT 
    train_data = np.concatenate((train_data,test_data), axis=0) 
    train_labels = np.concatenate((train_labels, test_labels), axis=0) 
    

    # ========================================================================== 
    # Create a new neural network from scratch. 
    # ==========================================================================
    if args.load == None:  
        print("Update: Indexing word vectors") 
        base_nn = create_base_nn_updated(create_embedding(word_index,glove_dir, 
                   max_nb_words))
        # Using the same input for both? As it seems to be just about dimensions
        # Creating the inputs # what does this line actually do? check mnist script 
        question_input = Input(shape=(max_seq_len,))
        answer_input = Input(shape=(max_seq_len,))
        q_nn = base_nn(question_input)  
        a_nn = base_nn(answer_input) 
        adam = Adam(lr=0.001) 
        distance = Lambda(euclidean_distance,
                output_shape=eucl_dist_output_shape)([q_nn,a_nn])

        model = Model([question_input, answer_input], distance)     
        # compile and fit
        model.compile(loss=contrastive_loss, optimizer=adam, metrics=['accuracy']) 
        history = model.fit([train_data[:,0], train_data[:,1]], train_labels, 
                batch_size=bs, epochs=epochs, validation_split=0, shuffle=True)    
        save_file = 'saved_models/epochs_' + str(epochs) + '_bs_'  + str(bs) + '.h5'
        model.save(save_file) 
        
        training_graph(history)       
    # ==========================================================================
    # Load a neural network 
    # ==========================================================================
    else: 
        print("Loading Model") 
        model = load_model(args.load, custom_objects={'contrastive_loss':contrastive_loss}) 
        history = model.fit([train_data[:,0], train_data[:,1]], train_labels,
                batch_size=32, epochs=0, validation_split=0.2)     
    # ==========================================================================
    # Evaluate and display results
    # ==========================================================================
    train_out = model.evaluate([train_data[:,0], train_data[:,1]] , train_labels, batch_size=32) 
    test_out = model.evaluate([test_data[:,0], test_data[:,1]] , test_labels, batch_size=32)  
    display_results(train_out, test_out, model)
    evaluation(sequences, model)  
