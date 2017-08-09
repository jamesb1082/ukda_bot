from __future__ import print_function
import glob 


"""
Simple class which acts as a wrapper to a python dictionary for the knowledge
and question banks.

Used by the bot to interface with either the knowledge bank or the question bank

"""
class DataManager():
    """
    Initialises a new instance of the class.

    Args: 
        data (str): The name of the directory where the data files are located. 
    """
    def __init__(self, data): 
        self.__kpath= '../data/' # MUST BE CHANGED AS HARDWIRED
        self.__kpath+=data
        self.__kpath+='/'
        self.__extension = ".txt" 
        self.__knowledge = {} 
        self.load_data() 
    
    def get_knowledge(self): 
        return self.__knowledge
    """
    Loads the data in the supplied folder into a dictionary. 
    """
    def load_data(self): 
        path = self.__kpath + "*" + self.__extension 
        for fname in glob.glob(path): 
            f1 = open(fname, 'r') 
            string =f1.read() 
            self.__knowledge[fname] = string.decode('utf8', 'ignore').encode('utf8')  

            f1.close() 
             
    """
    Gets the content of a file given a specific file name.

    Args:
        name (str): The name of the file one wants to retrieve.

    Returns: 
        str: The contents of the file as a string. 
    """
    def get_page(self, name):
        try:
            return self.__knowledge[self.__kpath + name + self.__extension] 
        except: 
            print("ERROR file not found: ",self.__kpath + name + self.__extension)
#            exit()

