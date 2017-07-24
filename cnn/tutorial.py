from keras.models import Sequential
from keras.layers import Dense 
import numpy as np 

np.random.seed(7) 

#load pima indians dataset 

dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",") 
X = dataset[:,0:8] 
Y = dataset[:,8]

# create model 

model = Sequential() 
model.add(Dense(12,input_dim=8, activation='relu') ) 
model.add(Dense(8,activation='relu')) 
model.add(Dense(1,activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

model.fit(X,Y,epochs=150,batch_size=10) 

#evaluate the model 
scores = model.evaluate(X,Y)
print() 
print("scores: %s" % scores) 
