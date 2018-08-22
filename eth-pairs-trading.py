# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:18:07 2018

@author: pxr13258
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm # 最小二乘
from statsmodels.stats.outliers_influence import summary_table 
#from functions import makeols
def zscore(series):
    return (series - series.mean()) / np.std(series)


pathbcc='D:/newdata/BCCUSDT_1H.csv'
bcc= pd.read_csv(pathbcc)
bccc=bcc['Close'].values
patheth='D:/newdata/ETHUSDT_1H.csv'
eth= pd.read_csv(patheth)
ethc=eth['Close'].values
para=1.45
delta=para*ethc-bccc
def zscore(series):
    return (series - series.mean()) / np.std(series)
def findzs(window,delta):
    zs=[]
    for i in range(window):
        zs.append(0)
    for i in range(window,len(delta)):
        current=delta[i-window+1:i+1]
        z=zscore(current)[-1]
        zs.append(z)
    return zs

acteth=[0]
amoeth=[para]
bar1=1.5
bar11=1.5
bar2=-1.5
bar21=-1.5
sta=-1
k=0
window=20
zs=findzs(window,delta)
for i in range(1,len(zs)):
    last=zs[i-1]
    if zs[i]>=bar1 and last<bar1:
        acteth.append(-1)
        amoeth.append(para)
        k=k+1
    elif zs[i]<bar11 and last>=bar11:
        acteth.append(1)
        amoeth.append(para)
        k=k+1
    elif zs[i]<bar2 and last>=bar2:
         acteth.append(1)
         amoeth.append(para)
         k=k+1
    elif zs[i]>=bar21 and last<bar21:
        acteth.append(-1)
        amoeth.append(para)
        k=k+1
    else:
        acteth.append(0)
        amoeth.append(para)
ethfinal=pd.DataFrame({'action':acteth,'amount':amoeth})
df2 = pd.concat( [eth, ethfinal], axis=1) 
df2.to_csv('ethpair.csv',index=False)   