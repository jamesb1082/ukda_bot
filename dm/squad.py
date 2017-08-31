from __future__ import division 
import json 
import pandas as pd 
class SquadManager(): 

    
    def __init__(self, data = "train-v1.1"): 
        self.__kpath = '../data/squad/' + data + '.json'
        self.__knowledge={} 
        
        self.__questions = [] 
        self.__answers = [] 
        self.load_data() 


    def load_data(self): 
        """
        Loads both questions and answers. 
        """
        with open(self.__kpath,'r') as f: 
            # data is a list of dicts 
            data = json.load(f)['data'] # gets rid of the version field. 
        answers = []  
        questions = [] 
        pairs = [] 
        for article in data: 
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']: 
                    questions.append(qa['question']) 
                    for a in qa['answers']:
                        answers.append(a['text'])
                        pairs.append((len(questions)-1, len(answers)-1)) 
        self.__questions = questions 
        self.__answers = answers 
        # Just checks that the qa pairs is the same length as the answers.  
       
        f = open("../data/squad/test_qa_pairs.csv", 'w')
        for p in pairs: 
            f.write(str(p[0])+"," + str(p[1])+",0\n") 
        f.close() 
    def print_facts(self): 
        print(len(self.__questions))
        print(len(self.__answers))
        print(len(self.__questions)/len(self.__answers))
        #print(qa['answers']) 
    

    def get_question(self, index):

        return self.__questions[int(index) ]


    def get_answer(self, index): 
         return self.__answers[int(index) ]


    def get_questions(self): 
        return self.__questions 


    def get_answers(self): 
        return self.__answers

if __name__ == '__main__': 
    dms = SquadManager("dev-v1.1")
 
