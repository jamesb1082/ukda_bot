from __future__ import print_function
import json 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
import numpy as np


def get_data():
    """
    Loads the data from json, however, it then has to be put into numpy matrix
    and generate the test label answers. Do this for if answer is in chunk.  
    """
    contexts = [
            "Warsaw", 
            "Victoria_(Australia)", 
            "Normans", 
            "Amazon_rainforest", 
            "Private_school"
            ]
    context_lists = []
    
    entries = []

    with open("dev-v1.1.json","r") as f: 
        data = json.load(f)['data'] 
    for context in contexts: 
        for entry in data: 
            if entry["title"] == context: 
                context_lists.append(entry["paragraphs"])
                break # used to exit forloop. Could change to while loop later
    
    for context in context_lists: 
        for row in context: 
            qas = row["qas"] 
            con = row["context"]
            for qa in qas: # currentyl doesn#'t split on ? 
                for c in con.split("."): 
                    entries.append(
                            (qa["question"].encode("utf-8"), 
                        c.encode("utf-8"), 
                        [x["text"].encode("utf-8") for x in qa["answers"]])
                        ) 
    return entries

def label(chunk,answers): 
    for ans in answers: 
        if ans in chunk:
            return 0 
    return 1 

def format_data(): 
    max_len = 1000 
    data = get_data()  
    labels = [] 
    arr = np.zeros((len(data), 2,max_len)) 
    tokenizer = Tokenizer(num_words=10000,
            filters = '#$%()*+,-./:;?@[zz]^_{|}~\t\n',lower=True, split=" ")
    
    text_list = [] 

    for x in data: 
        text_list.append(x[0]) 
        text_list.append(x[1]) 
        for y in x[2]:
            text_list.append(y) 
    tokenizer.fit_on_texts(text_list)  
    i = 0 
    for row in data: 
        sequences = tokenizer.texts_to_sequences([row[0],row[1]])  
        sequences = pad_sequences(sequences,max_len) 
        arr[i][0] = sequences[0]
        arr[i][1] = sequences[1] 
        i += 1
        labels.append(label(row[1], row[1])) 

    return arr, labels, tokenizer #could include tokenizer if needed. 





