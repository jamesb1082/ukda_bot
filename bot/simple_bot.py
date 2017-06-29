from __future__ import print_function 
from __future__ import division 
from chatterbot import ChatBot 
from chatterbot.trainers import ListTrainer 
from data_manager import Data_manager 
import sys
from time import time 


class Simple_bot:
    """
    Constructor for the chatbot. Sets up the relevant adapters and loads up data managers to 
    handle the knowledge banks. 
    """
    def __init__(self): 
        self.__q_bank = Data_manager('questions') 
        self.__k_bank = Data_manager('knowledge') 
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
                database= "data/database.db", 
                )
        self.__chatbot.set_trainer(ListTrainer)        
        self.train()
        print("Chatbot created")
        print("----------------------------------------")
        print("Ask the bot a question to get a response") 
        print("----------------------------------------")
    
    def load_conversations(self, q,k): 
        """
        Loads conversations from the knowledge base using a CSV file which has a 
        question and an answer.

        Args:
            q (Data_manager): This is the datamanager for the questions. 
            k (Data_manager): This is the dtaamanager for the knowledge base. 
        """
        path = "data/qa.csv"
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
        """
        conversations = self.load_conversations(self.__q_bank, self.__k_bank)  
        i = 0 
        t0 = time() 
        for conversation in conversations: 
            i+=1 
            try:
                self.__chatbot.train(conversation) 
            except KeyError: 
                print("error has occurred on line " , i, "in qa.csv")
               # print(conversation[1])

        print("trained on", i, "examples in", round(time()-t0, 3), "s") 

    def run(self):
        """
        Runs the chatbot in terminal mode.
        """
        while True: 
            try: 
                self.__bot_input = self.__chatbot.get_response(None) 
            except(KeyboardInterrupt, EOFError, SystemExit): 
                break 
            print("=================================================================")        




#reload(sys) 
#sys.setdefaultencoding('utf8')

#print("----------------------------------------")
#bot1 = Simple_bot() 

#bot1.run() 
