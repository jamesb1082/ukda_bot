from __future__ import print_function 
import re
"""
Usage Notes: 
    As each webpage in the CSV dump had a contents spanning multiple lines
    each of which contains commas. So they were seperated into each webpage
    having it's own dedicated txt file with its contents. To do this, 
    START had to be put into every entry (when editing the CSV in libre office) so to be able to distinguish which multi-line entry belonged to each webpage. 

    example: 
    START
    al1
    al2
    START
    bl1
    bl2
    bl3
    START
    cl1
    START
    ...
"""

chars = {
    '\xc2\x82' : ',',        # High code comma
    '\xc2\x84' : ',,',       # High code double comma
    '\xc2\x85' : '...',      # Tripple dot
    '\xc2\x88' : '^',        # High carat
    '\xc2\x91' : '\x27',     # Forward single quote
    '\xc2\x92' : '\x27',     # Reverse single quote
    '\xc2\x93' : '\x22',     # Forward double quote
    '\xc2\x94' : '\x22',     # Reverse double quote
    '\xc2\x95' : ' ',
    '\xc2\x96' : '-',        # High hyphen
    '\xc2\x97' : '--',       # Double hyphen
    '\xc2\x99' : ' ',
    '\xc2\xa0' : ' ',
    '\xc2\xa6' : '|',        # Split vertical bar
    '\xc2\xab' : '<<',       # Double less than
    '\xc2\xbb' : '>>',       # Double greater than
    '\xc2\xbc' : '1/4',      # one quarter
    '\xc2\xbd' : '1/2',      # one half
    '\xc2\xbe' : '3/4',      # three quarters
    '\xca\xbf' : '\x27',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : '',          # modifier - under line
    '\xe2\x80\x93' : "-", 
    '\xe2\x80\x99': "'", 
    '\xe2\x80\x94':''
}
def replace_chars(match):
    """
    Replaces the characters.    
    """
    char = match.group(0)
    return chars[char]

def edit(text):
    """
    Returns a new string with any banned characters replaced. 
    """
    return re.sub('(' + '|'.join(chars.keys()) + ')', replace_chars, text)


###############################################################################
# Main program starts here
###############################################################################
content = open("data/website_prepper/content.csv", 'r')  
title = open("data/website_prepper/title.csv", 'r')

contents = [] 
titles = []
output = ''
for line in content:
    line = line.replace(',,,,,,', '')
    line = line.replace('\\n', ' ') 
    line = line.replace('"', '')
    line = line.replace('&amp', ' and ')
    line = line.strip() 
    if line == 'START':
        contents.append(edit(output)) 
        output = '' 
    else: 
        output+=line
    

for row in title:
    row = row.strip()
    row = row.lower()  
    row = 'data/knowledge/w_' + row 
    row+='.txt' 
    titles.append(row) 
print("----error checking, titles and contents should be same no-----") 
print("titles:", len(titles)) 
print("contents:", len(contents))
for i in range(0, len(titles)): 
    nf = open(titles[i], 'w')  
    nf.write(contents[i])  
    nf.close() 


content.close() 
title.close()
print("Files generated") 
