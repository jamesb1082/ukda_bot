from data_manager import DataManager
from random import randint, shuffle 
import progressbar 
import sys 
def get_data(): 
    """
    Gets data from the new dataset. 

    Returns: 
        a list of tuples in the form of (question index, answer index, label) 
        a list of strings
    """
    index_links = [] # do list of tuples (q,a,l) where q and a is index in rawstrings
    #links = get_file_links("../data/debug_dataset.csv" )  
    links = get_file_links() 
    texts = get_raw_strings() 
    dmq = DataManager("questions") 
    dma = DataManager("knowledge") 
    strings = [] 
    
    count = 0 
    bar = progressbar.ProgressBar(maxval=len(links),
            widgets=["Loading dataset: ", progressbar.Bar('=','[',']'), ' ', 
                progressbar.Percentage(), ' ',  
                progressbar.ETA()])
    bar.start()
    i = 0 
    for row in links:
        current = []
        for i in range(0,len(texts)):
            if dmq.get_page(row[0]) == texts[i]: 
                current.append(i)
                break
        for i in range(0,len(texts)):
            if dma.get_page(row[1]) == texts[i]: 
                current.append(i)
        index_links.append((current[0], current[1], int(row[2])))
        count+=1
        bar.update(count)
    sys.stdout.write("\n") 
    return index_links, texts

def get_raw_strings(): 
    """
    Loads all the strings in the dataset.

    Returns a list of strings
    """
    texts = [] 
    dmq = DataManager("questions")
    dma = DataManager("knowledge")

    for key, item in dmq.get_knowledge().items():
        texts.append(item)
    for key, item in dma.get_knowledge().items(): 
        texts.append(item) 
    return texts


def get_file_links(data_dir = '../data/new_qa.csv'):  
    data = [] 
    f = open(data_dir, 'r') 
    for line in f: 
        data.append(line.strip("\n").split(","))
    f.close() 
    return data 


def create_dataset(number = 2):
    """
    Creates a new dataset which maps 1 right answer and every other answer as wrong.
    Notdone for all possible answers, but all the other correct answers. 
    ask spyros if this should be changed. 
    """
    dataset = [] 
    correct_ans = [] 
    qa_path = "../data/qa.csv"
    new_dataset = "../data/new_qa.csv" 
     
    #read in the right answer questions.
    qa_file = open(qa_path,'r') 
    for i in range(0,number):  
        entry= qa_file.readline().strip("\n").split(",") 
        correct_ans.append(entry) 
    qa_file.close() 
     
    for ans in correct_ans: 
        for other_ans in correct_ans: 
            if other_ans[1] != ans[1]: 
               # Adds 1 right answer and 1 wrong answer for every wrong answer. 
                dataset.append([ans[0],ans[1],0])
                dataset.append([ans[0], other_ans[1],1])
    shuffle(dataset) 
    f = open(new_dataset, 'w') 
    
    for row in dataset: 
        entry = row[0] + ',' + row[1] + ',' + str(row[2])+"\n" 
        f.write(entry)
    f.close()
 


























