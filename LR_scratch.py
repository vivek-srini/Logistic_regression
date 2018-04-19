#Logistic Regression
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values.T
Y=dataset.iloc[:,-1].values.reshape(-1,1).T
X=(X-np.mean(X))/np.std(X)
#Splitting the dataset into training set and test set


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)


def initialize_weights(n_x):
    W=np.zeros((n_x,1))
    b=np.zeros((1,1))
    return W,b

def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A

def forward_prop(X,W,b):
    Z=np.dot(W.T,X)+b
    A=sigmoid(Z)
    return A

def compute_cost(A,Y):
    m=Y.shape[1]
    cost=(-1/m)*(np.sum(np.multiply(np.log(A),Y)+np.multiply(np.log(1-A),1-Y)))
    return cost

def backward_prop(X,Y,A):
    m=Y.shape[1]
    dZ=A-Y
    dW=np.dot(X,dZ.T)/m
    db=np.sum(dZ)/m
    return dW,db

def Grad_Desc(W,b,dW,db,learning_rate):
    W=W-learning_rate*dW
    b=b-learning_rate*db
    return W,b

def logistic_model(X,Y,learning_rate,num_iter):
    W,b=initialize_weights(X.shape[0])
    for i in range(1,num_iter):
        
        A=forward_prop(X,W,b)
        dW,db=backward_prop(X,Y,A)
        W,b=Grad_Desc(W,b,dW,db,learning_rate)
        if i%100==0:
            print(compute_cost(A,Y))
    return W,b,dW,db
def predict(X,W,b):
    A=forward_prop(X,W,b)
    prediction=(A>0.5)
    return prediction
W,b,dW,db=logistic_model(X,Y,0.0025,15000)   
prediction=predict(X,W,b)
count=0
for i in range(1,Y.shape[1]):
    if prediction[0][i]==Y[0][i]:
        count+=1
accuracy=count/float(Y.shape[1])