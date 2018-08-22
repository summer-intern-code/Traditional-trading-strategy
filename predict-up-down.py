# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:01:51 2018

@author: pxr13258
"""

# %load prob1.py
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
# Load dataset
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras import optimizers


data = pd.read_csv("D:/dataset/LTCBTC_15T.csv",index_col= 'DetailTime')
#data.head(5)
#print(ts)
train=[]
test=[]
test2=[]
test3=[]

def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back),0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back,0]) 
    return np.array(dataX), np.array(dataY)
#for i in range(1000):
    #temp=data[len(data)-288*2-i-1:len(data)-288-i]
    #temp.index = pd.to_datetime(temp.index)
    #train.append(temp)
    #temp1=data[len(data)-288-i:len(data)-288-i+1]
    #temp1.index = pd.to_datetime(temp1.index)
    #test.append(temp1)
    #temp2=data[len(data)-288-i-20:len(data)-288-i+80]
    #temp2.index = pd.to_datetime(temp2.index)
    #test2.append(temp2)
    #temp3=data[len(data)-288-i-20:len(data)-288-i]
    #temp3.index = pd.to_datetime(temp3.index)
    #test3.append(temp3)25627
for i in range(300):
    #emp=data[8111+i:8111+i+1000]
    temp=data[26900+i:26900+i+300]
    temp.index = pd.to_datetime(temp.index)
    train.append(temp)
   #temp1=data[8111+i+1000:8111+i+1001]
    temp1=data[26900+i+300:26900+i+301]
    temp1.index = pd.to_datetime(temp1.index)
    test.append(temp1)
   #temp2=data[8111+i+1000-20:8111+i+1000+50]
    temp2=data[26900+i+300-20:26900+i+300+50]
    temp2.index = pd.to_datetime(temp2.index)
    test2.append(temp2)
real80=[]
test80=[]
real1=[]
pre1=[]
pre11=[]
reallast=[]

    


#writer.writerow(['pre1','real1','reallast'])
#for i in range(m):

    #writer.writerow([pre1[i],real1[i],reallast[i]])

#csvFile2.close()

for i in range(300):
    print(i)
    source1=train[i]
    target=test2[i]
    ts = source1['Close']
    ts1=target['Close']
    only1=test[i]['Close'][0]
    real1.append(only1)
    dataset=[]
    testset=[]
    for i in range(len(ts)):
        dataset.append(ts[i])
    for i in range(len(ts1)):
        testset.append(ts1[i])
    dataset=np.array(dataset)
    testset=np.array(testset)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset=dataset.reshape(-1,1)
    testset=testset.reshape(-1,1)
#print(dataset)
    #dataset = scaler.fit_transform(dataset)
    #testset=scaler.fit_transform(testset)


    look_back = 20
    trainX, trainY = create_dataset(dataset, look_back)
    testX,testY= create_dataset(testset, look_back)
#print(trainX.shape[1])
   
    y_trainnorm=trainY-trainX[:,-1]
    y_testnorm=testY-testX[:,-1]
    one=np.ones((len(trainX[:,1]),1))
    X=np.column_stack((one,trainX))
    xtran=np.transpose(X)
    xdot=np.dot(xtran,X)
    xinv=np.linalg.inv(xdot)
    xddot=np.dot(xinv,xtran)
    w=np.dot(xddot,y_trainnorm)

    one=np.ones((len(testX[:,1]),1))
    Xt=np.column_stack((one,testX))
    #delta=np.dot(Xt,w)-testY
    #testPredict = scaler.inverse_transform([np.dot(Xt,w)]) 
    testPredict =np.dot(Xt,w)
    #trainPredict= scaler.inverse_transform([np.dot(X,w)])
    trainPredict=  np.dot(X,w)
    #testY = scaler.inverse_transform([y_testnorm])
    pre1.append(testPredict[0])
    reallast.append(ts[-1])
    print(testPredict[0],float(only1-ts[-1]))
    