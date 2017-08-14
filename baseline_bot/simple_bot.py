from __future__ import print_function 
from __future__ import division 
from chatterbot import ChatBot 
from chatterbot.trainers import ListTrainer 
import sys
sys.path.insert(0,"../") # used to find packages the same level 
from time import time 
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import subprocess 
from dm.managers import DataManager 
class Simple_bot:
    """
    Constructor for the chatbot. Sets up the relevant adapters and loads up data managers to 
    handle the knowledge banks. 
    """
    def __init__(self, ratio, testing=True): 
        self.__tr = float(ratio) 
        self.__q_bank = DataManager('questions') 
        self.__k_bank = DataManager('knowledge') 
        
        if testing:
           self.__chatbot = ChatBot("UKDA Bot", 
                    storage_adapter="chatterbot.storage.JsonFileStorageAdapter", 
                    logic_adapters=[
                        #                    "chatterbot.logic.MathematicalEvaluation", 
                        #                    "chatterbot.logic.TimeLogicAdapter", 
                        "chatterbot.logic.BestMatch"
                        ], 
                    input_adapter="chatterbot.input.VariableInputTypeAdapter",
                    silence_performance_warning=True,  
                    output_adapter="chatterbot.output.OutputAdapter", 
                    database= "../data/database.db", 
                    )

        else: 
            self.__chatbot = ChatBot("UKDA Bot", 
                    storage_adapter="chatterbot.storage.JsonFileStorageAdapter", 
                    logic_adapters=[
                        #                    "chatterbot.logic.MathematicalEvaluation", 
                        #                    "chatterbot.logic.TimeLogicAdapter", 
                        "chatterbot.logic.BestMatch"
                        ], 
                    input_adapter="chatterbot.input.TerminalAdapter",
                    silence_performance_warning=True,  
                    output_adapter="chatterbot.output.TerminalAdapter", 
                    database= "../data/database.db", 
                    )
        self.__convos = self.load_conversations(self.__q_bank, self.__k_bank) 
        self.__chatbot.set_trainer(ListTrainer)        
        self.train()
        print("Chatbot created")
    
    def load_conversations(self, q,k): 
        """
        Loads conversations from the knowledge base using a CSV file which has a 
        question and an answer.

        Args:
            q (Data_manager): This is the datamanager for the questions. 
            k (Data_manager): This is the datamanager for the knowledge base. 
        """
        path = "../data/qa.csv"
        conversations = [] 
        csvfile = open(path, 'r') 
        for line in csvfile:
            stripped = line.strip("\n") 
            value = stripped.split(",")
            conversation = [q.get_page(value[0]), k.get_page(value[1])]  
            conversations.append(conversation) 
        csvfile.close() 
        return conversations 


    def train(self):
        """
        Trains the chatbot on the questions and answers loaded from it's CSV file. 
        By changing the conversations to a class variable, the program has sped up 
        by factor of 10. 
        """
        value = int(len(self.__convos) * self.__tr) 
        conversations = self.__convos[:value]
        i = 0 
        t0 = time() 
        for c in range(1):
            for conversation in conversations: 
                i+=1
                try:
                    self.__chatbot.train(conversation) 
                except KeyError: 
                    print("error has occurred on line " , i, "in qa.csv")
                   # print(conversation[1])
        print("trained on", i, "examples in", round(time()-t0, 3), "s") 

    def __build_corpus(self, conversations): 
        corpus = [] 
        for c in conversations: 
            corpus.append(c[0]) 
        return corpus
    def test(self): 
        value = int(len(self.__convos) * self.__tr) 
        conversations = self.__convos[value:]
        corpus = self.__build_corpus(conversations) 
        n_features = 2000 
        vectorizer = CountVectorizer(min_df =1) 
        X = vectorizer.fit_transform(corpus)
        print(X)
    def test2(self):         
        value = int(len(self.__convos) * self.__tr) 
        conversations = self.__convos[value:]
        tp= 0 
        t0 = time()
        i = 0
        correct = 0
        for c in conversations:
            i+=1
            question = c[0] 
            answer = c[1]
            guess = str(self.__chatbot.get_response(question))
            fname = '../data/output/'+str(i)+'.txt'
            f = open(fname, 'w') 
            f.write(guess+"/n"+answer) 
            f.close() 
            if guess == answer:
                correct+=1
                
        print("tested on", len(conversations), "examples in", round(time()-t0, 3), "s") 
        print("Accuracy: ", round(tp/len(conversations)*100,2), "%")
        print("True positives: ", tp) 
            



    def chat(self):
        """
        Runs the chatbot in terminal mode.
        """
        while True: 
            try: 
                self.__bot_input = self.__chatbot.get_response(None) 
            except(KeyboardInterrupt, EOFError, SystemExit): 
                break 
            print("=================================================================")


if __name__ =='__main__': 
    try:  
        subprocess.call("./rmdatabase.sh", shell=False)     
    except IOError: 
        print("Data base file not found") 

    reload(sys) 
    sys.setdefaultencoding('utf8')
    parser = argparse.ArgumentParser("Runs the chatbot in either training or test mode") 
    parser.add_argument("-t","--test", help="run in test mode", action="store_true")
    
    args = parser.parse_args() 
    #print(argp1[1]) 
    print("---------------------------------------")
    if args.test:
        print("Mode: test") 
        print("Training please wait...") 
        bot1 = Simple_bot(0.8)
        print("Testing please wait...") 
        bot1.test2()

    else:
        print("Mode: chat")
        print("Training please wait...") 
        bot1 = Simple_bot(1, False)
        print("Type a message and press enter to get a response") 
        bot1.chat()     



