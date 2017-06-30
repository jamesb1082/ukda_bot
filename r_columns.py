def r_columns(l1):
    l2 = [l1[i] for i in range(len(l1)) if i in keep] 
    return l2

def two_d(matrix): 
    keep = []
    new = [] 
    for i in range(len(matrix[0])): #flip through the columns
        for r in matrix: #gets each row
            if r[i] > 0: 
                keep.append(i)  
    for row in matrix:
        new.append( [row[i] for i in range(len(row)) if i in keep]) 
    return new 
m = [[0,1,0],[1,0,0],[0,1,0]] 
print(m)
m = [[0,2,0]]
m2 = two_d(m)
print(m2) 
