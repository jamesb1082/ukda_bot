from __future__ import print_function
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout, Input, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation
from keras.models import Sequential, Model, load_model
import os 
import sys 
import numpy as np 
from preprocess import get_data, get_file_links, get_raw_strings
from keras import backend as K 
from keras.utils import plot_model
from sklearn.metrics import classification_report 
import seaborn as sns 
import pandas
import argparse 
from data_manager import DataManager

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

    return train_data,test_data,train_labels,test_labels, word_index, data 

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


def get_answers(question, answers, model, num_ans=1): 
    """
    Gets the answers with the closest distance
    """
    score_rating = [] # corresponds to the score given by predict between question and answer.
    question = question.reshape(1,2300)
    for answer in answers:  
        answer = answer.reshape(1,2300)  
        score_rating.append(model.predict([question,answer]) ) 
    top_val = 10000
    top_pos = 10000
    for i in range(0,len(score_rating) ): 
        if score_rating[i] <  top_val: 
            top_val = score_rating[i] 
            top_pos = i
    return (top_val, top_pos)  

def evaluation(sequences): 
    """
    evaluates the neural network as a distance function 
    """ 
    dma = DataManager("knowledge") 
    dmq = DataManager("questions") 
    links = get_file_links("../data/qa.csv") # index of thte question and answer in the texts list. 
    index_links = [] 
    texts = get_raw_strings() 
    for row in links : 
        current = [] 
        for i in range(0, len(texts)): 
            if dmq.get_page(row[0]) == texts[i]: 
                current.append(i) 
        for i in range(0, len(texts)): 
            if dma.get_page(row[1]) == texts[i]: 
                current.append(i) 
        index_links.append(current) 

    answers = sequences[171:] #we know that the number of questions is 171. This is hardcoded.  
    questions = sequences[:171] 
    # build labels 
    labels = [] 
    for row in index_links: 
        labels.append(row[1])
    prediction = []  
    for question in questions: 
        ans = get_answers(question, answers, model) 
        prediction.append((172 + ans[1]))  
    print(classification_report(labels, prediction) )  
    for i in range(0, len(labels)): 
        print(labels[i] , " = ", prediction[i]) 

if __name__ == '__main__':  
    parser = argparse.ArgumentParser("Siamese neural network for question answering")
    parser.add_argument("-l", "--load", help="Load a neural network", 
            action="store", type=str) 
    args = parser.parse_args() 
    # variables 
    glove_dir = '../../vectors/glove/' 
    max_seq_len = 2300 
    max_nb_words = 200000
    embedding_dim = 100 
    validation_split = 0.2 # what does valiation split mean? 
    save_file = 'saved_models/snn.h5'

    print("Update: Preprocessing data") 
    train_data, test_data, train_labels, test_labels, word_index, sequences= load_data() 
    

    if args.load == None:  
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
                trainable=False)

        base_nn = create_base_nn_updated(embedding_layer)
        # Using the same input for both? As it seems to be just about dimensions
        q_nn = base_nn(question_input)  
        a_nn = base_nn(answer_input) 

        distance = Lambda(euclidean_distance,
                output_shape=eucl_dist_output_shape)([q_nn,a_nn])

        model = Model([question_input, answer_input], distance)

        # compile and fit
        model.compile(loss=contrastive_loss, optimizer="Adam", metrics=['accuracy']) 
        history = model.fit([train_data[:,0], train_data[:,1]], train_labels, 
                batch_size=32, epochs=1000, validation_split=0.2) 
    
       # plot some graphs 
        dt  = history.history['acc'] 
        data = pandas.DataFrame({"acc":dt})  
        ax = sns.tsplot(data=data["acc"] )
        ax.set(xlabel="epoch", ylabel="Accuracy") 
        sns.plt.show()      
    
    
    
    
    
    else: # loads a model from saved file 
        print("Loading Model") 
        model = load_model(args.load, custom_objects={'contrastive_loss':contrastive_loss}) 
        history = model.fit([train_data[:,0], train_data[:,1]], train_labels, 
                batch_size=32, epochs=100, validation_split=0.2) 
    
 
    # Predict and evaluate.
    train_out = model.evaluate([train_data[:,0], train_data[:,1]] , train_labels, batch_size=32) 
    test_out = model.evaluate([test_data[:,0], test_data[:,1]] , test_labels, batch_size=32) 
    model.save(save_file)  
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
    print("Update: Evaluating")  
    evaluation(sequences)  
