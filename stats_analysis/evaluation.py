from __future__ import print_function, division 
import sys
sys.path.insert(0, "../") 
from dm.managers import DataManager
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from wmd_python.wmdqa import WMDQA
from sklearn.metrics import classification_report 
import numpy as np 
import json 

def load_corpus(q,k): 
    path = "../data/qa.csv" 
    corpus = [] 
    csvfile = open(path,'r') 
    for line in csvfile:
        stripped = line.strip("\n") 
        value = stripped.split(",") 
        pair = [q.get_page(value[0]), k.get_page(value[1])] 
        corpus.append(pair)  
    csvfile.close() 
    return corpus


def load_knowledge(k): 
    corpus = []
    for value in k.get_knowledge(): 
        f = open(value, 'r') 
        corpus.append(f.read()) 
        f.close() 
    return corpus 


def return_ranked(q, ans, k_corpus): 
    """
    Returns list where each entry contains cosine sim and index pos of k_corpus

    """
    
    sims = [] 
    for a in ans: 
        sims.append(cosine_similarity(q.reshape(1,-1) ,a.reshape(1,-1)))
    
    idxs = [x for x in range(0,len( k_corpus))] 
    joint_ans = zip(sims, idxs)
    highest = -5
    ans = 0       
    for row in joint_ans: 
        if row[0] > highest: 
            highest = row[0] 
            ans = row 
    return ans[1]
     
def report(answers, pred, true): 
    # t_names = ["Class " + str(x) for x in range(0, len(answers))]
    #print(classification_report(true,pred,target_names=t_names)) 
    
    print(true)
    print(classification_report(true,pred)) 

def find_pos(ans, k_corpus):   
    for i in range(0, len(k_corpus)):
        if ans == k_corpus[i]: 
            return i  
    return -1 

def method_a(): 
    #load files and build  corpus
    q_bank = DataManager('questions') 
    k_bank = DataManager('knowledge') 
    corpus = load_corpus(q_bank, k_bank)
    print(corpus[0]) 
    pred_dict = {} 
    # split questions and answers into two seperate lists
    q_corpus = [] 
    k_corpus = load_knowledge(k_bank)   
    real_ans = [] 
    true_ans = [] 
    qa_dicts = []  
    
    i = 0
    for c in corpus: 
        q_corpus.append(c[0])
        real_ans.append(c[1])
        qa_dicts.append({'id':str(i) , 'answers':{ 'text' : c[1]} })
        i+=1
    overall_dict = {'version' : '1.1', 'data':qa_dicts}  
    
    with open('../data/datafileA.json', 'w')  as outfile:
            json.dump(overall_dict, outfile) 


    for ans in real_ans: 
        true_ans.append(find_pos(ans,k_corpus)) 
    corpus = q_corpus+ k_corpus 
    print("real ans:") 
    print(true_ans)
    #set up and create vectorizer 
    vectorizer = TfidfVectorizer(min_df = 0.01, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    #split the vectors up to questions and answers. 
    questions = X.toarray()[:len(q_corpus)]
    answers = X.toarray()[len(q_corpus):]
    output = [] 
    
    
    for i in range(0, len(q_corpus)): 
        pred = return_ranked(questions[i], answers, k_corpus) 
        output.append(pred) 
   #     pred_dict[str(i)] = k_corpus[pred].replace('\n', ' ').replace('\r', ' ')   
        pred_dict[str(i)] = k_corpus[pred]          
    
    print("predicted ans:") 
    print(output) 
    with open('../data/predictedA.json', 'w') as outfile: 
        json.dump(pred_dict, outfile) 
    report(k_corpus, output,true_ans)


def method_b():
    ans = []
    #load files and build  corpus
    q_bank = DataManager('questions') 
    k_bank = DataManager('knowledge') 
    corpus = load_corpus(q_bank, k_bank)
    # split questions and answers into two seperate lists
    q_corpus = [] 
    k_corpus = load_knowledge(k_bank)   
    true = [] 
    qa_dicts = [] 
    corpus = corpus[1:] 
    i = 0 
    pred_dict = {}  
    for c in corpus: 
        q_corpus.append(c[0]) 
        true.append(find_pos(c[1], k_corpus)) 
        qa_dicts.append({'id':str(i), 'answers':{'text' : c[1]}})
        i+=1 
    wmdqa = WMDQA()
    data = k_corpus
    overall_dict = {'version' : '1.1', 'data':qa_dicts}  
 
    with open('../data/datafileB.json', 'w')  as outfile:
        json.dump(overall_dict, outfile) 

    i = 0  
    for row in corpus: 
        q = row[0]  
#        print(q)
        data = k_corpus
        top_mems = wmdqa.getTopRelativeMemories(q,data,1)
        data = np.array(data)[top_mems]
        ans.append(find_pos(data,k_corpus))
        print(data) 
        print("==============================================") 
        pred_dict[str(i)] = data.tolist()[0]
        i+=1 
        #ans.append((find_pos(data[0], k_corpus),find_pos(data[1],k_corpus), find_pos(data[2],k_corpus)))
    with open('../data/predictedB.json', 'w') as outfile: 
        json.dump(pred_dict, outfile) 

    print("predicted ans: ") 
    print(ans)
    print()
    print("real ans:") 
    print(true)
    report(0, ans,true)      

print("METHOD A") 
print("======================================") 
method_a()
print()
print()
print("METHOD B") 
print("======================================") 
method_b() 
