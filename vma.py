# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:12:01 2018

@author: pxr13258
"""
#VMA t =RV t−1 ∗Close t−1 +(1−RV t−1 )∗VMA t−1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from functions import dm,truerange,di,findtheta
patheth='D:/newdata/ETHBTC_1H.csv'
eth= pd.read_csv(patheth)
ethc=eth['Close'].values
ethh=eth['High'].values
ethl=eth['Low'].values
ethv=eth['TotalVolume'].values
def meanv(datav,days,today):
    return np.mean(datav[today+1-days:today+1])
def highlowv(meanvol,days,today):
    return max(meanvol[today+1-days:today+1]),min(meanvol[today+1-days:today+1])
def rv(hav,lav,av):
    return (av-lav)/(hav-lav)
def vma(closelast,rvlast,vmalast):
    return rvlast*closelast+(1-rvlast)*vmalast
N=6
currentstate=-1
sell=[]
buy=[]
ethmv=[]
for i in range(N):
    ethmv.append(ethv[i])
for i in range(N,len(ethc)):
    ethmv.append(meanv(ethc,N,i))
#ethav0=meanv(ethc,N,N)
#ethhav0,ethlav0=highlowv(ethav0,N,N)
#ethrv0=rv(ethhav0,ethlav0,ethav0)
ethvmalast=ethc[N]
for i in range(N+2,len(ethc)):
    ethav=ethmv[i-1]
    ethhav,ethlav=highlowv(ethmv,N,i-1)
    ethrv=rv(ethhav,ethlav,ethav)
    ethvma=vma(ethc[i-1],ethrv,ethvmalast)
    #ethvmalast=ethvma
    if ethc[i-1]<ethvma and ethvma<ethvmalast and currentstate==-1:
        currentstate=1
        buy.append(i)
    if ethc[i-1]>ethvma and currentstate==1:
        currentstate=-1
        sell.append(i)
        if i-buy[-1]>=3:
           plt.plot(ethc[buy[-1]:i])
           plt.show()
    ethvmalast=ethvma