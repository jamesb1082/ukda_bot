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
    links = get_file_links("../data/debug_dataset.csv" )  
 
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
    print(len(texts)) 
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


def create_dataset(repeat=3):
    """
    Creates a random dataset in the format: question, answer,correct
    Order is also random. 
    """
    dataset = [] 
    correct_ans = [] 
    qa_path = "../data/qa.csv"
    new_dataset = "../data/new_qa.csv"  
    qa_file = open(qa_path, 'r') 
    # read data in the qa.  
    for line in qa_file:
        entry = line.strip("\n").split(",")  
        correct_ans.append(entry) 
    qa_file.close() 

    for ans in correct_ans: 
        for i in range(repeat): 
            # generates random wrong answers. 
            dataset.append([ans[0], ans[1],1]) 
            value = randint(0, len(correct_ans)-1)             
            row = correct_ans[value]
            # ensures that a wrong answer is picked every time. 
            if row[1] == ans[1]: 
                while row[1] == ans[1]: 
                    value = randint(0,len(correct_ans)-1) 
                    row = correct_ans[value] 

            dataset.append([ans[0],row[1], 0 ])  
    print(len(dataset)) 
    print(len(correct_ans))
    shuffle(dataset) 
    f = open(new_dataset, 'w') 
    
    for row in dataset: 
        entry = row[0] + ',' + row[1] + ',' + str(row[2])+"\n" 
        f.write(entry)
    f.close()


a,b = get_data() 
i = 0 
for row in a: 
    if row[0] > 276:
        i+=1 
print(i) 
#print(a) 
