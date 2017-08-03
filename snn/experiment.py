from __future__ import print_function 
import numpy as np 
import evaluate as e 
import preprocess as pp 

# stored the numpy sequences from last model. 

"""
Simple mock up of a possible approach to persistance.

Loads the sequences from the stored model. 

It also generates the naswer index and can be used to retrieve an answer from a model prediction. 
"""
arr = np.load("sequences.npz.npy")

links = pp.get_file_links("../data/new_qa.csv") 
texts = pp.get_raw_strings() 
index_links, corr_in_links = e.generate_index(links,texts) 
relevant_ans, answer_indexes = e.ans_map(index_links, arr) 


