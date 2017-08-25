from __future__ import print_function 
import numpy as np 
import evaluate as e
import preprocess as pp 
import pickle 
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from unidecode import unidecode
from keras.models import load_model 
from train_bot import contrastive_loss, build_model 
from dm.squad import SquadManager

class Chatbot(): 
    def __init__(self):
        """
        initialises the chatbot
        """
        links = pp.get_file_links("../data/squad/new_qa.csv") 
        texts = pp.get_raw_strings() 
        self.texts1 = texts  
        index_links, self.__corr_in_links = e.generate_index(links,texts) 
        
        self.__tokenizer = self.get_tokenizer() 
        
        sequences = self.get_sequences(self.__tokenizer, texts) 
        
        self.__relevant_ans, self.__answer_indexes = e.ans_map(index_links, sequences)  
        self.__word_index = self.__tokenizer.word_index
        
        dms = SquadManager() 
        self.__text_answers = dms.get_answers()   
        print(self.__text_answers[1]) 
        self.__seq_answers = sequences[len(dms.get_questions()):] 
        #self.__text_answers = texts[276:]  # in evaluate this is actually sequences. 
        
        
        # =========================================================================
        # Load model and weights
        # =========================================================================
        self.__model = build_model(self.__word_index, 100)
        self.__model.load_weights("models/weights.hdf5") 
       # self.__model = load_model("saved_models/saved_model.h5", 
       #         custom_objects={"contrastive_loss":contrastive_loss})
        self.__model._make_predict_function()
    
    
    def get_tokenizer(self):
        """
        Loads a keras tokenizer which was fitted when training model using  pickle 
        """
        with open('models/tokenizer.p', 'r') as f: 
            tokenizer = pickle.load(f)  
        return tokenizer

    
    def get_sequences(self, tokenizer,texts):
        """
        generates padded sequences from a prefitted tokenizer. 
        """
        sequences = self.__tokenizer.texts_to_sequences(texts) 
        return pad_sequences(sequences,2300) 
     
    
    def get_answer(self, question): 
        """
        Gets the answer for a specific question 
        """
        #q = question.encode('utf8')  
        q_sequence = self.__tokenizer.texts_to_sequences([unidecode(question)])
        q_sequence = pad_sequences(q_sequence,2300) 
        answer = e.get_answers(q_sequence, self.__relevant_ans, self.__model,"",
                self.__answer_indexes) 
        #        return self.__text_answers[answer[1]
        
        #print( self.texts1[self.__answer_indexes[answer[1]]]) 
        #print(self.__text_answers[answer[1]]) 
        return self.__text_answers[self.__answer_indexes[answer[1]]] 
        #return self.texts1[self.__answer_indexes[answer[1]]]
    
    
    def interactive(self): 
        """
        This is interactive mode, continually ask questions and answer them. For use in
        terminal. 
        """
        while True: 
            print() 
            print()    
            new_question = raw_input("Please enter your data management question: ")     
            print()  
            # loses words that it hasn't seen before. 
            print("==============================Answer==============================")
            print()
            print(self.get_answer(new_question))  


if __name__ == '__main__': 
    c = Chatbot()
    c.interactive() 
