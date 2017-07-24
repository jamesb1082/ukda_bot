from keras.preprocessing.text import Tokenizer

s = ["My name is james andrew brill", "wow this is cool"] 

tokenizer = Tokenizer(num_words=1000) 
tokenizer.fit_on_texts(s) 
word_index = tokenizer.word_index
i = 0 
for word in word_index: 
    i+=1 
    print(word)

print(i) 
