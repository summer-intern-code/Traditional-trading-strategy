# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:42:13 2018

@author: pxr13258
"""

#阻力支撑
#最高价，最低价都超过上一个阶段

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pathbcc='D:/newdata/LTCBTC_1H.csv'
bcc= pd.read_csv(pathbcc)
detailtime=list(bcc['DetailTime'].values)
bccc=bcc['Close']
bcch=bcc['High']
bccl=bcc['Low']

import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import summary_table # 
from functions import makeols,ols
#window=6
#bar=2.07#1.5
#bar1=-0.7


plt.plot(bccc)
win=24
buy=[]
sell=[]
stoploss=0.98
stopprofit=1.1
def makeindex(series):
    number=0
    num=[]
    for i in range(len(series)):
        num.append(number)
        number=number+1
    return num
num=makeindex(bccc)
for i in range(win+1,len(bccc)-1):
    past=bccc[i-win-1:i]
    now=bccc[i-win:i+1]
    if max(past)<max(now) and min(past)<min(now):
        buy.append(i)
    if max(past)>max(now) and min(past)>min(now):
        sell.append(i)
for i in range(len(buy)):
    plt.scatter(buy[i],bccc[buy[i]],color='red')
#for i in range(len(sell)):
#    plt.scatter(sell[i],bccc[sell[i]],color='green')

def endup(k,start,bccc,sell):
    if bccc[k]<=stoploss*max(bccc[start:k]):
            return k
        
    if bccc[k]/bccc[k-3]>stopprofit:#or bccc[i]/bccc[start]>1.05:
            return k
    if k in sell:
            return k
    return 0
sta=-1
last=0
cul0=0
cul=[]
for i in range(len(bccc)):
    if sta==-1:
        if i in buy:
            last=i
            sta=1
    elif sta==1:
        if endup(i,last,bccc,sell)==i:
            sta=-1
#            plt.plot(num[last:i+1],bccc[last:i+1],color='red')
            print(last,i,bccc[i]-bccc[last])
            cul0=cul0+bccc[i]-bccc[last]
            cul.append(cul0)
#plt.plot(cul)           
plt.show()





















window=24
window=5
bar=1.55
bar1=-1.5
#bar=2
#bar1=-1#-1.05
#2.07,-0.7
red=[]
green=[]
beta,re,re1=ols(window,bccc,bccl,bcch)
#re-----re1
for i in range(len(re)):
    rightdev=re[i]
    if rightdev>=bar:
       red.append(i+2*window)
#       plt.scatter(i+window,bccc[i+window],color='red')  
    if rightdev<=bar1:
       green.append(i+2*window)
#       plt.scatter(i+window,bccc[i+window],color='green')  
sta=-1
sell=[]
buy=[]
actlist=[]
amountlist=[]
for i in range(window):
    actlist.append(0)
    amountlist.append(1)
for i in range(window,len(bccc)):
    if sta==-1 and i in red:
        buy.append(i)
        sta=1
        actlist.append(1)
        amountlist.append(1)
    elif sta==1 and i in green:
        sell.append(i)
        sta=-1
        actlist.append(-1)
        amountlist.append(1)
    else:
        actlist.append(0)
        amountlist.append(1)
cul=0
for i in range(len(sell)):
    print(buy[i],sell[i])
    plt.plot(bccc[buy[i]:sell[i]],color='red')
    cul=cul+bccc[sell[i]]-bccc[buy[i]]
#    plt.show()
print(cul)
bccfinal=pd.DataFrame({'action':actlist,'amount':amountlist})
df = pd.concat( [bcc, bccfinal], axis=1) 
df.to_csv('rsrsltc.csv',index=False)          
    