import glob 


"""
Simple class which acts as a wrapper to a python dictionary for the knowledge
and question banks.

Used by the bot to interface with either the knowledge bank or the question bank

"""
class Data_manager():
    """
    Initialises a new instance of the class. 
    """
    def __init__(self, data): 
        self.__kpath= 'data/' # MUST BE CHANGED AS HARDWIRED
        self.__kpath+=data
        self.__kpath+='/'
        self.__extension = ".txt" 
        self.__knowledge = {} 
        self.load_data() 

    """
    loads the data in the supplied folder into a dictionary. 
    """
    def load_data(self): 
        path = self.__kpath + "*" + self.__extension 
        for fname in glob.glob(path): 
            f1 = open(fname, 'r') 
            string =f1.read() 
            self.__knowledge[fname] = string.decode('utf8', 'ignore').encode('utf8')  

            f1.close() 
             
    """
    gets the content of a file given a specific file name. 
    """
    def get_page(self, name):
        return self.__knowledge[self.__kpath + name + self.__extension] 


