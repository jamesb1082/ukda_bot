from __future__ import print_function 
import numpy as np 
import evaluate as e 
import preprocess as pp 
import pickle 
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# stored the numpy sequences from last model. 

def get_tokenizer():
    """
    Loads a keras tokenizer which was fitted when training model using  pickle 
    """
    with open('tokenizer.p', 'r') as f: 
        tokenizer = pickle.load(f)  
    return tokenizer

def get_sequences(tokenizer,texts):
    """
    generates padded sequences from a prefitted tokenizer. 
    """
    sequences = tokenizer.texts_to_sequences(texts) 
    return pad_sequences(sequences,2300) 
    
def run(): 
    """
    Get a question as user input and return an answer.     
    """
    links = pp.get_file_links("../data/new_qa.csv") 
    texts = pp.get_raw_strings() 
    index_links, corr_in_links = e.generate_index(links,texts) 
    tokenizer = get_tokenizer() 
    sequences = get_sequences(tokenizer, texts) 
    relevant_ans, answer_indexes = e.ans_map(index_links, sequences) 
    new_question = raw_input("Please enter your datamanagement question: ")     


"""
Simple mock up of a possible approach to persistance.

"""
run() 
