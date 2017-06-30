from  __future__ import print_function, division
from tabulate import tabulate 
from copy import deepcopy
import glob 

class ConfusionM: 
    def __init__(self):
        self.__headers, self.__names = self.__get_headers() 
        self.__grid = self.__build_empty(self.__headers)

    
    def __build_empty(self, headers):
        """
        Builds an empty matrix using the headers to gauge the size. 
        """
        return [[0 for x in range(len(headers))] for y in range(len(headers))] 

    def __get_headers(self): 
        """
        Gets all the file names from the knowledge bank. 
        Stores the headers as numbers and provides a dictionary for referencing purposes.
        Saves on space when printing it out. 
        """
        name_dict = {} 
        path = "data/knowledge/*.txt"
        headers = []
        i = 0 
        for fname in glob.glob(path): 
            fname = fname.split("/") 
            fname = fname[2].split(".") 
            headers.append(i)
            name_dict[i] = fname[0]
            i+=1 
        return headers, name_dict 

    def __dimension_reduction(self,matrix, h): 
        """
        Takes a list not a python dictionary. 

        This performs dimension reduction mainly for output. 
        """
        new_matrix = [] #create the new matrix. 
        h2 = [] # store the new headers

        for row in matrix:
            rest= row[1:]
            if sum(rest) > 0:
                new_matrix.append(row)              
        #return column_reduction(new_matrix, h)
        return self.__column_reduction(new_matrix, h)  

    def __column_reduction(self, matrix, headers): 
        """
        Gets rid of all zero columns. 
        """
        keep = [] 
        new = [] 
        for i in range(0, len(headers)):
            found = False
            for r in matrix: 
                if r[i] > 0 and found == False: 
                    keep.append(i) 
                    found = True
        for row in matrix: 
            new.append([row[i] for i in range(len(row)) if i in keep])
        
        
        new_keep = [i-1 for i in keep if i > 0] 
        h = [headers[i] for i in range(len(headers)) if i in new_keep]

        return new, h 

    def __get_value(self, x,y): 
        row = self.__grid[x] 
        return row[y]

    def insert(self, row,column, value):
        """
        Inserts a value into a x,y location of the matrix. 
        """
        r =  self.__grid[row] 
        r[column] = value

    def display(self):
        """
        Prints the matrix. 

        Performs dimensionality reduction on the length axis if the number of 
        headers is above a certain size.
        """
        grid = [] 
        grid = deepcopy(self.__grid)
        new_headers= [] 
        new_headers=deepcopy(self.__headers) 
        for i in range(len(new_headers)):
            grid[i].insert(0,str(new_headers[i])+"|")
        
        if len(self.__headers) >10: 
            print("reduction")
            grid, new_headers= self.__dimension_reduction(grid, new_headers) 
        new_headers.insert(0, '') 
        f = open("wow.txt", 'w')
        f.write(tabulate(grid,new_headers))
        f.close()
        print(tabulate(grid, new_headers)) 

    def num_ref(self):
        print(self.__headers)
        for i in self.__headers: 
            print(i,": ", self.__names[i]) 
        

    def __precision(self, col): 
        """
        TP is where row is the same as class.
        Precision is calculated by the TP / sum of column.

        """
        tp = self.__get_value(col, col) 
        total = 0 
        for i in range(0, len(self.__grid)): 
            total+= self.__get_value(i, col)
        if total==0: 
            return 0 
        return tp/total 
    def overall_precision(self):
        """
        Average precision accross all labels. 
        """
        value = 0 
        for i in range(0, len(self.__grid)): 
            value+=self.__precision(i) 
        return value/len(self.__grid) 
    
    def __recall(self, row): 
        """
            Sums up the values in each row. 
        """
        return sum(row) 
    def overall_recall(self):
        """
            Average accross all rows. 
        """
        value = 0 
        for row in self.__grid: 
            value += self.__recall(row) 
        return value/len(self.__grid) 
        

c = ConfusionM()

for i in range(0, 5,2):
    c.insert(i,i,1)
c.display()
print(c.overall_precision()) 
print(c.overall_recall()) 
