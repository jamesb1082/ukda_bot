from data_manager import DataManager
from preprocess import get_file_links, get_raw_strings
from sklearn.metrics import classification_report

def generate_index(links, texts):
    """
    Returns two list of lists. Which contain reference points one index for both q and a. 
    """
    dma = DataManager("knowledge") 
    dmq = DataManager("questions") 

    index_links = []
    correct_index_links = [] # for the correct answer pairs 
    for row in links: 
        current = [] 
        for i in range(0, len(texts)): 
            if dmq.get_page(row[0]) == texts[i]: 
                current.append(i) 
        for i in range(0, len(texts)): 
            if dma.get_page(row[1]) == texts[i]: 
                current.append(i) 
        index_links.append(current)
        if int(row[2]) == 0: #assuming 0 is the label for correct answers. 
            correct_index_links.append(current) 

    return index_links, correct_index_links


def generate_predictions(questions, corr_links, relevant_ans, model,ans_dict): 
    """
    Returns list of predictions using get answer() function 
    """
    
    pred = [] 
    for row in corr_links: 
        ans = get_answers(questions[row[0]], relevant_ans, model, row[1],ans_dict)
        pred.append( ans_dict[ans[1]]) 
    return pred




def get_answers(question,answers,model,correct_ans,converter): 
    """
    get the answers with the closest distance
    """
    score_rating = [] 
    question = question.reshape(1,2300)

    for i in range(len(answers)):
        answer = answers[i].reshape(1,2300) 
        score_rating.append(model.predict([question,answer]))
        #score_rating.append(dummy_distance(converter[i], correct_ans))
        top_val = 100
        top_pos = -1
     
    for i in range(0,len(score_rating)): 
        if score_rating[i] < top_val: 
            top_val = score_rating[i]
            top_pos = i 
    
    a = [converter[x] for x in range(len(answers))] 
    print(score_rating) 

    print(a) 
    return (top_val, top_pos) 




def dummy_distance2(answer,index): 
    if answer==index:
        return 0
    return 1



def evaluation(sequences, model): 
    links = get_file_links("../data/debug_dataset.csv")
    texts = get_raw_strings() 
    index_links, corr_in_links = generate_index(links, texts)  
    
    answers = sequences[276:] 
    questions = sequences[:276]
    
    #get labels for each question. Uses correct ans links to get right labels only. 
    labels = [] 
    for row in corr_in_links: 
        labels.append(row[1]) 

    relevant_ans = [] 
    answer_indexes = {} 
    for li in index_links: 
        if any((sequences[li[1]]==x).all() for x in relevant_ans) == False: 
            relevant_ans.append(sequences[li[1]])
            answer_indexes[len(relevant_ans)-1] =li[1] 

    prediction = generate_predictions(questions,corr_in_links, 
            relevant_ans, model, answer_indexes)     
    
    for i in range(0, len(prediction)): 
            print("label: ", labels[i], "    prediction: ", prediction[i]) 
    #print(classification_report(labels,prediction))  

