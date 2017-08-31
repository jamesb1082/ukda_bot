import sys
sys.path.insert(0,"../") 
from random import randint, shuffle 
import progressbar 
import sys 
from dm.squad import SquadManager 
from unidecode import unidecode 


def get_data(): 
    """
    Gets data from the new dataset (squad). 
    Returns: 
        a list of tuples in the form of (question index, answer index, label) 
        a list of strings
    """
    index_links = [] # do list of tuples (q,a,l) where q and a is index in rawstrings
    #links = get_file_links("../data/debug_dataset.csv" )  
    links = get_file_links() 
    texts = get_raw_strings() 
    dmsquad = SquadManager() 
    
    strings = [] 
    
    # blocked out as it is time consuming and not needed due to file format. 
    """ 
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
            if dmsquad.get_question(row[0]) == texts[i]: 
                current.append(i)
                break
        for i in range(0,len(texts)):
            if dmsquad.get_answer(row[1]) == texts[i]: 
                current.append(i)
        index_links.append((current[0], current[1], int(row[2])))
        count+=1
        bar.update(count)
    """
    sys.stdout.write("\n")
    index_links  = links 
    return index_links, texts


def get_raw_strings(dev_set = False): 
    """
    Loads all the strings in the dataset.

    Returns a list of strings
    """
    texts = []

    if dev_set == True: 
        dmsquad = SquadManager("dev-v1.1") 
    else: 
        dmsquad= SquadManager() 
    for  item in dmsquad.get_questions():
        texts.append(unidecode(item))  
    
    for item in dmsquad.get_answers(): 
        texts.append(unidecode(item))        

    return texts


def get_file_links(data_dir = '../data/squad/new_qa.csv'):  
    data = [] 
    f = open(data_dir, 'r') 
    for line in f: 
        data.append(line.strip("\n").split(","))
    f.close() 
    return data 


def estimate_rows(n):
    return n * n * 2 



def split_dataset(p_split, numq):
    qa_path = "../data/squad/qa.csv" 
    qa_file = open(qa_path,'r') 
    training_path = "../data/squad/train_correct.csv" 
    test_path = "../data/squad/test_correct.csv" 
    correct_ans = []  
    counter = 0  
    for line in qa_file:
        counter+=1 
        entry = line.strip("\n").split(",")
        correct_ans.append(entry)
        if counter > numq: #stops it being created on the whole dataset.  
            break 

    qa_file.close()


    value = int(len(correct_ans) * (1- p_split)) 
    train_questions = correct_ans[:value] 
    test_questions = correct_ans[value:]  
    
    with open(training_path, 'w') as f: 
        for row in train_questions: 
            entry = row[0] + ',' + row[1]+"\n" 
            f.write(entry)
    
    with open(test_path, 'w') as f: 
        for row in test_questions:
            #note they have to have a 0 on the end to fit with the evaluation script
            entry = row[0] + ',' + row[1]+"," + str(0)  + "\n" 
            f.write(entry)
    

    


def create_dataset(number = 2):
    """
    Creates a new dataset which maps 1 right answer and every other answer as wrong.
    Notdone for all possible answers, but all the other correct answers. 
    ask spyros if this should be changed. 
    """
    dataset = [] 
    correct_ans = [] 
    qa_path = "../data/squad/train_correct.csv"
    new_dataset = "../data/squad/new_qa.csv"

    # for the squad database 
    
    print("Estimated number of rows: ", estimate_rows(number))    
    #read in the right answer questions.
    qa_file = open(qa_path,'r') 
    for i in range(0,number):  
        entry= qa_file.readline().strip("\n").split(",") 
        correct_ans.append(entry) 
    qa_file.close() 
    
    
    count = 0 
    bar = progressbar.ProgressBar(maxval=len(correct_ans),
            widgets=["Generating Pairs: ", progressbar.Bar('=','[',']'), ' ', 
                progressbar.Percentage(), ' ',  
                progressbar.ETA()])
    bar.start()
    count = 0 
    for ans in correct_ans: 
        for other_ans in correct_ans: 
            if other_ans[1] != ans[1]: 
               # Adds 1 right answer and 1 wrong answer for every wrong answer. 
                dataset.append([ans[0],ans[1],0])
                dataset.append([ans[0], other_ans[1],1])
    
        count+=1
        bar.update(count)
    sys.stdout.write("\n")  
    shuffle(dataset)  
    f = open(new_dataset, 'w') 
    bar = progressbar.ProgressBar(maxval=len(dataset),
            widgets=["Storing pairs: ", progressbar.Bar('=','[',']'), ' ', 
                progressbar.Percentage(), ' ',  
                progressbar.ETA()])
    bar.start()
    count = 0 
    for row in dataset: 
        entry = row[0] + ',' + row[1] + ',' + str(row[2])+"\n" 
        f.write(entry)
        count+=1 
        bar.update(count) 
    sys.stdout.write("\n")      
    f.close()



def create_sqds(number , start ):
    """
    Creates a new dataset which maps 1 right answer and every other answer as wrong.
    Notdone for all possible answers, but all the other correct answers. 
    ask spyros if this should be changed. 
    
    This can only be used to create the full dataset using a batching technique.
    do not use this to create a partial dataset instead us create_dataset(number)  
    """
    dataset = [] 
    correct_ans = [] 
    qa_path = "../data/train_correct.csv"
    new_dataset = "../data/new_qa.csv"

    # for the squad database 
    qa_path = "../data/squad/qa.csv"
    new_dataset = "../data/squad/new_qa.csv"
    
    
    #read in the right answer questions.
    qa_file = open(qa_path,'r')
    for line in qa_file: 
        entry= line.strip("\n").split(",") 
        correct_ans.append(entry) 
    qa_file.close() 
    count = 0

    num_batch = 349
    bar = progressbar.ProgressBar(maxval=num_batch,
            widgets=["Generating Pairs: ", progressbar.Bar('=','[',']'), ' ', 
                progressbar.Percentage(), ' ',  
                progressbar.ETA()])
    bar.start()
    start = 0
    finish = 0 
    batch_size = 251
    
    
    for i in range(0, num_batch): 
        dataset = []
        start = finish
        finish = start+batch_size
        for j in range(start, finish): 
            ans = correct_ans[j]  
            for other_ans in correct_ans: 
                if other_ans[1] != ans[1]: 
                   # Adds 1 right answer and 1 wrong answer for every wrong answer. 
                    dataset.append([ans[0],ans[1],0])
                    dataset.append([ans[0], other_ans[1],1]) 
                
        shuffle(dataset)  
        f = open(new_dataset, 'a') 
        for row in dataset: 
            entry = row[0] + ',' + row[1] + ',' + str(row[2])+"\n" 
            f.write(entry)
        f.close()
        del dataset 
        count+=1
        bar.update(count)
    sys.stdout.write("\n")  
        

