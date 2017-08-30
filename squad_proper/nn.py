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
from keras import backend as K
import seaborn as sns 
import pandas
import argparse
import matplotlib.pyplot as plt 
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle 
from data_loader import format_data

def create_base_nn_updated(embedding): 
    """
    Same as the above function, however it is deeper.  
    """
    
    #return nn_2(embedding) 
    filters = 256 
    kernel_size = 3
    d_value = 0.1 
    seq = Sequential() 
    seq.add(embedding)
    seq.add(Dropout(d_value))
    seq.add(Conv1D(filters, kernel_size,padding='same',activation='elu', strides=1))
  #  seq.add(Flatten())  
    seq.add(GlobalMaxPooling1D()) 
    for i in range(2):
        seq.add(Dense(512))
        seq.add(Dropout(d_value))  
        seq.add(Activation('elu'))
        seq.add(Dropout(d_value))    
    seq.add(Dense(1))
    seq.add(Activation('linear')) 
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


def create_embedding(word_index, glove_dir, max_nb_words, embedding_dim, max_seq_len):
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


def build_model(word_index, embedding_dim=100): 
    """
    Returns the siamese neural network model. 
    """
    glove_dir = '../vectors/glove/' 
    max_nb_words = 200000 
    max_seq_len = 1000 # hard  coded into the length of the strings
    base_nn = create_base_nn_updated(create_embedding(word_index,glove_dir, 
                   max_nb_words, embedding_dim, max_seq_len)) 
    
    question_input = Input(shape=(max_seq_len,))
    answer_input = Input(shape=(max_seq_len,))
    q_nn = base_nn(question_input)  
    a_nn = base_nn(answer_input) 
    distance = Lambda(euclidean_distance,
                output_shape=eucl_dist_output_shape)([q_nn,a_nn])
    
    return Model([question_input, answer_input], distance)  


def training_graph(history): 
    """
    Displays a graph showing acc vs epochs
    """
    dt  = history.history['acc'] 
    data = pandas.DataFrame({"acc":dt})      
    ax = sns.tsplot(data=data["acc"] )
    ax.set(xlabel="epoch", ylabel="Accuracy") 
    plt.show() 


if __name__ == '__main__':  
    parser = argparse.ArgumentParser("Siamese neural network for question answering")
    parser.add_argument("-l", "--load", help="Load a neural network", 
            action="store", type=str) 
    args = parser.parse_args() 
    
    
    # ==========================================================================
    # variables 
    # ==========================================================================
    validation_split = 0.2 
    save_file = 'models/snn.h5'
    epochs = 10
    bs = 64#batch size  
    max_seq_len = 1000
    embedding_dim = 100 
    # ==========================================================================
    # Pre-process the data
    # ==========================================================================
    print("Update: Preprocessing data") 
    
    train_data, train_labels, word_index = format_data()  

    # ========================================================================== 
    # Create a new neural network from scratch. 
    # ==========================================================================
    if args.load == None:  
        print("Update: Indexing word vectors") 
        # Using the same input for both? As it seems to be just about dimensions
        # Creating the inputs # what does this line actually do? check mnist script 
        adam = Adam(lr=0.001) 
        
        model = build_model(word_index) 
        # compile and fit
        model.compile(loss=contrastive_loss, optimizer=adam, metrics=['accuracy']) 
        
        checkpointer = ModelCheckpoint("models/weights.hdf5", verbose=1,
                save_best_only=True) 

        tb = TensorBoard(log_dir='./Log', histogram_freq=0, write_graph=True,
                write_images=True)
        print(train_data.shape) 
        history = model.fit([train_data[:,0], train_data[:,1]], train_labels, 
                batch_size=bs, epochs=epochs, validation_split=0.2, shuffle=True, 
                callbacks=[checkpointer, tb])    
        model.load_weights("models/weights.hdf5")     
        
        save_model= 'models/epochs_' + str(epochs) + '_bs_'  + str(bs) + '.h5'
        model.save(save_model)
    
        training_graph(history)       
    # ==========================================================================
    # Load a neural network 
    # ==========================================================================
    else: 
        print("Loading Model") 
        model = load_model(args.load, custom_objects={'contrastive_loss':contrastive_loss}) 
        #history = model.fit([train_data[:,0], train_data[:,1]], train_labels,
        #        batch_size=32, epochs=0, validation_split=0.2)     
