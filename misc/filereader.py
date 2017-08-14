"""
    Used to quickly display the contents of one of the question files without having to search for it manually. 
"""


from __future__ import print_function 

while True:
    print("========================================") 
    a = str(input(("enter the file number: "))) 
    print()  

    fname = 'data/questions/qthelp-' + a + '.txt' 
    f = open(fname, 'r') 
    print(f.read()) 
    f.close()
