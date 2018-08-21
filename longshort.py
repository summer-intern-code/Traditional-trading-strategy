# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:14:32 2018

@author: pxr13258
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from functions import dm,truerange,di,findtheta
pathbcc='D:/newdata/LTCBTC_1H.csv'
bcc= pd.read_csv(pathbcc)
detailtime=list(bcc['DetailTime'].values)
bccc=bcc['Close']
bcch=bcc['High']
bccl=bcc['Low']
def makeindex(series):
    number=0
    num=[]
    for i in range(len(series)):
        num.append(number)
        number=number+1
    return num
num=makeindex(bccc)
r_ewma=12
maa=24
bccc_r=pd.ewma(np.array(bccc),r_ewma)
bcch_r=pd.ewma(np.array(bcch),r_ewma)
bccl_r=pd.ewma(np.array(bccl),r_ewma)
bccdmp,bccdmm=dm(bcch_r,bccl_r)
bcctr=truerange(bccc_r,bcch_r,bccl_r)
bccdip,bccdim,bccdx,bccadx=di(bccdmp,bccdmm,bcctr)

bccc=np.array(bccc)
ma_bccc=pd.rolling_mean(bccc, maa)
sd_bccc=pd.rolling_std(bccc, maa) 
top_bccc=ma_bccc+2*sd_bccc 
bot_bccc=ma_bccc-2*sd_bccc


godownindex=[]
def goup(k,bccc,top_bccc,sd_bccc,bccadx):
     if bccc[i]>top_bccc[k]*theta and findtheta(bccc[k-thetaback:k])>=bolling_bar and sd_bccc[k]>=bolling_sd and bccadx[k]>=adxbar:
        return k
     return 0
def godown(k,bccc,bot_bccc,sd_bccc,bccadx):
    if bccc[k]<bot_bccc[k]*(2-theta1) and findtheta(bccc[k-thetaback1:k])<=-bolling_bar1 and sd_bccc[k]>=bolling_sd1 and bccadx[k]>=adxbar1:
        return k
    return 0
def enddown(k,start,bccc,top_bccc):
    if bccc[k]<=stoploss1*bccc[start]: #min(bccc[start+1:i]):
           return k
    if bccc[k]>=top*top_bccc[k]:
            return k
    notinup=0
    for j in range(start+3,k):
            if godown(k,bccc,bot_bccc,sd_bccc,bccadx)==k:
                notinup=0
            else:
                notinup=notinup+1
    if notinup>=back1:
            return k
    if goup(k,bccc,top_bccc,sd_bccc,bccadx)==k:
        return k
    if bccc[start]/bccc[k]>stopprofit1:#or bccc[i]/bccc[start]>1.05:
            return k
    return 0

#up and down
#up parameter
theta=0.97
back=48
thetaback=14
bolling_bar=0.15
bolling_sd=0.0002
end_sd_bar=0
adxbar=70
bot_bar=1

stopprofit=1.1#2
stoploss=0.93
theta1=0.97
#down parameter
back1=49#56
thetaback1=4#24
dopoint=[]
bolling_bar1=0.15#0
bolling_sd1=0.0002#0.0007
theta1=0.95#0.998
adxbar1=70#80
stopprofit1=1.04
stoploss1=0.94
top=1



def endup(k,start,bccc,top_bccc,sd_bccc,bccadx):
    notinup=0
    if bccc[k]<=stoploss*max(bccc[start:k]):
            return k
        
    if bccc[k]/bccc[k-3]>stopprofit:#or bccc[i]/bccc[start]>1.05:
            return k
    notinup=0
    for j in range(start+3,k):
        if bccc[j]>top_bccc[j]*theta and findtheta(bccc[j-thetaback:j])>=bolling_bar and sd_bccc[j]>=bolling_sd and bccadx[j]>=adxbar:
                notinup=0
        else:
                notinup=notinup+1
    if notinup>=back:
            return k
    if bccc[k]<=bot_bccc[k]*bot_bar:
            return k
    return 0
sta=[-1,-1]
lastbuy=0
lastsell=0
cul1=0
cull1=0
cul=[]
cull=[]
win=0
actlist=[]
amountlist=[]
startpoint=20
plt.plot(bccc)
for i in range(startpoint):
       actlist.append(0)
       amountlist.append(0)
for i in range(startpoint,len(bccc)-1):
    if sta==[-1,-1]:
        if goup(i,bccc,top_bccc,sd_bccc,bccadx)==i:
            lastbuy=i
            sta=[1,-1]
            actlist.append(1)
            amountlist.append(1)
        elif godown(i,bccc,bot_bccc,sd_bccc,bccadx)==i:
            lastsell=i
            sta=[1,1]
            actlist.append(-1)
            amountlist.append(1)
        else:
            actlist.append(0)
            amountlist.append(1)
    elif sta==[1,-1]:
        if  endup(i,lastbuy,bccc,top_bccc,sd_bccc,bccadx)==i:
            print(lastbuy,i,'up',bccc[i]-bccc[lastbuy]*1.001)
            cul1=cul1+bccc[i]-bccc[lastbuy]*1.001
            cull1=cull1+bccc[i]-bccc[lastbuy]*1.001
            if (bccc[i]-bccc[lastbuy]*1.001)>=0:
                win=win+1
            plt.plot(num[lastbuy:i],bccc[lastbuy:i],color='red')
#            plt.show()
            cul.append(cul1)
            cull.append(cull1)
            sta=[-1,-1]
            lastbuy=0
            actlist.append(-1)
            amountlist.append(1)
        else:
            actlist.append(0)
            amountlist.append(1)
    elif sta==[1,1]:
         if enddown(i,lastsell,bccc,top_bccc)==i and  goup(i,bccc,top_bccc,sd_bccc,bccadx)!=i:
             print(lastsell,i,'down',bccc[lastsell]-bccc[i]*1.001)
             plt.plot(num[lastsell:i],bccc[lastsell:i],color='green')
#             plt.show()
             cul1=cul1+bccc[lastsell]-bccc[i]*1.001
             if (bccc[lastsell]-bccc[i]*1.001)>=0:
                 win=win+1
             cul.append(cul1)
             cull.append(cull1)
             sta=[-1,-1]
             lastsell=0
             actlist.append(1)
             amountlist.append(1)
         elif enddown(i,lastsell,bccc,top_bccc)==i and goup(i,bccc,top_bccc,sd_bccc,bccadx)==i:
             print(lastsell,i,'down',bccc[lastsell]-bccc[i]*1.001)
             plt.plot(num[lastsell:i],bccc[lastsell:i],color='green')
#             plt.show()
             cul1=cul1+bccc[lastsell]-bccc[i]*1.001
             if (bccc[lastsell]-bccc[i]*1.001)>=0:
                 win=win+1
             cul.append(cul1)
             cull.append(cull1)
             lastbuy=i
             lastsell=0
             sta=[1,-1]
             actlist.append(2)
             amountlist.append(1)
         else:
              actlist.append(0)
              amountlist.append(1)
    else:
         actlist.append(0)
         amountlist.append(1)
plt.show()
print(cul[-1],cull[-1],float(win/len(cul)))             
plt.plot(cul)
plt.plot(cull)
bccfinal=pd.DataFrame({'action':actlist,'amount':amountlist})
df = pd.concat( [bcc, bccfinal], axis=1) 
df.to_csv('bcc.csv',index=False) 