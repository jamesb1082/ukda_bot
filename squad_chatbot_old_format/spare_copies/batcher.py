from itertools import islice 
from preprocess import get_raw_strings
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np  


def num_per_epoch(batch_size): 
    with open("../data/squad/new_qa.csv") as f: 
        return sum([1 for line in f])/ batch_size 


def load_batches( tokenizer, n, epochs) :
    """
    doesn't strip the lines, some preprocessing will needed to be done 
    
    Currently only works for batches of 16 lols. Create dynamic code tomorrow.
    

    Getting rnadom errrors where it is awlays fialing on the last epoch for some reason.

    It is sometthing to do with the batch step is not the same WHY IS THIS 
    """
    max_nb_words = 200000
    texts = get_raw_strings()
    
    # Various generating the texts among other stuff  
    sequences = tokenizer.texts_to_sequences(texts) 
    word_index = tokenizer.word_index
    max_seq_len = 2300 
    data = pad_sequences(sequences,max_seq_len) 
    
    
    with open('tokenizer.p', 'w') as f: 
        pickle.dump(tokenizer,f) 
    
    labels = [] 
    
    for i in range(0, epochs+1): 
        with open("../data/squad/new_qa.csv") as f: 
            end = False
            while end == False :  
                batch =  list(islice(f,n)) 
                labels  = [] 
                if len(batch) == 0 : 
                    end = True 
                else:
                    arr = np.zeros((len(batch), 2, max_seq_len)) 
                    # strip and split each line as you loop through. Saves iteration
                    for i in range(0, len(batch)): 
                        row = batch[i].strip("\n").split(",")  
                        arr[i][0] = data[int(row[0])] 
                        arr[i][1] = data[int(row[1])] 
                        labels.append(row[2])
                    yield [arr[:,0], arr[:,1]], np.array((labels))              

"""
max_nb_words = 200000
t = Tokenizer(num_words=max_nb_words, 
        filters='#$%()*+,-./:;<=>?@[\\]^_{|}~\t\n', lower=True, split=" ")

t.fit_on_texts(get_raw_strings()) 
a = load_batches(t, 32, 2)
print(num_per_epoch(32))
print(sum(1 for b in a))"""
