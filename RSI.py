# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 16:19:43 2018

@author: pxr13258
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
patheth='D:/newdata/ETHBTC_1H.csv'
eth= pd.read_csv(patheth)
ethc=eth['Close']
def RSI(point,data,days):
    up=0
    down=0
    current=data[point-days:point+1].values
    for i in range(1,len(current)):
        if current[i]>=current[i-1]:
            up=up+(current[i]-current[i-1])
        else:
            down=down+current[i-1]-current[i]
    return float(up/(up+down))
rsi6=[]
rsi12=[]
delta=[]
day=6
day1=12
for i in range(day1+1,len(ethc)):
    rsi6.append(RSI(i,ethc,day))
    rsi12.append(RSI(i,ethc,day1))
    delta.append(RSI(i,ethc,day)-RSI(i,ethc,day1))
up=[]
for i in range(len(delta)):
    if delta[i]>=0.4:
        up.append([i+day1+1,ethc[i+day1+1]])
up=np.array(up)
plt.plot(ethc)
plt.scatter(up[:,0],up[:,1],color='red')
