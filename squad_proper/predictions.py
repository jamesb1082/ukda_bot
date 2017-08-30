import json 
import numpy as np 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_answer(model,tokenizer ,question, context): 
    answers = context.split(".")


    question = pad_sequences(tokenizer.texts_to_sequences([question]), 1000)[0]  
    
    converted_answers = pad_sequences(tokenizer.texts_to_sequences(answers), 1000)
    question = question.reshape(1,1000) 
    score_rating = [] 

    for i in range(len(answers)) : 
        answer  = converted_answers[i].reshape(1,1000) 
        score_rating.append(model.predict([question,answer]))
        top_val = 1000
        top_pos = -1 


        for i in range(0, len(score_rating)): 
            if score_rating[i] < top_val: 
                top_val = score_rating[i] 
                top_pos = i 
    return answers[top_pos]



def generate_predictions(model, tokenizer):
    results = {}  
    with open("dev-v1.1.json", 'r') as f: 
        data = json.load(f)['data'] 
        for article in data: 
            for paragraph in article['paragraphs']: 
                context = paragraph['context']  
                for qa in paragraph['qas']: 
                    results[qa['id']] = get_answer(model,tokenizer,
                            qa['question'].encode("utf-8"), 
                            context.encode("utf-8"))

    with open("pred.json", 'w') as f: 
        json.dump(results,f) 
