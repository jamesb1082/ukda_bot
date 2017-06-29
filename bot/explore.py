from __future__ import print_function, division
import pandas as pd  

def get_contents(fname): 
    f1 = open(fname, 'r') 
    temp_data = [] 
    for row in f1: 
        row = row.strip("\n") 
        row = row.split(",") 
        temp_data.append(row[1])
    f1.close()  
    return temp_data 

def generate_counts(data): 
    counts = []
    names=[]
    for row in data:
        if row not in names and row !='unknown': 
            names.append(row) 
            counts.append(1) 
        else:
            for i in range(len(names)): 
                if row == names[i]:
                    counts[i]+=1 

    return pd.DataFrame({"names":names, "count":counts}) 
