# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:48:04 2018

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

theta=0.95
thetaback=14
bolling_bar=0.14
bolling_sd=0.00023
adxbar=75

theta1=0.95
thetaback1=24#24
bolling_bar1=0.15#0
bolling_sd1=0.0003#0.0007
adxbar1=50#80



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
upindex=[]
upvalue=[]
doindex=[]
dovalue=[]
plt.plot(bccc)
def goup(k,bccc,top_bccc,sd_bccc,bccadx):
     if bccc[i]>top_bccc[k]*theta and findtheta(bccc[k-thetaback:k])>=bolling_bar and sd_bccc[k]>=bolling_sd and bccadx[k]>=adxbar:
        return k
     return 0
def godown(k,bccc,bot_bccc,sd_bccc,bccadx):
    if bccc[k]<bot_bccc[k]*(2-theta1) and findtheta(bccc[k-thetaback1:k])<=-bolling_bar1 and sd_bccc[k]>=bolling_sd1 and bccadx[k]>=adxbar1:
        return k
    return 0
for i in range(thetaback,len(bccc)):
    if goup(i,bccc,top_bccc,sd_bccc,bccadx)==i:
        upindex.append(num[i])
        upvalue.append(bccc[i])
for i in range(thetaback1,len(bccc)):
    if godown(i,bccc,bot_bccc,sd_bccc,bccadx)==i and goup(i,bccc,top_bccc,sd_bccc,bccadx)!=i:
        doindex.append(num[i])
        dovalue.append(bccc[i])
plt.scatter(upindex,upvalue,color='red')     
#plt.scatter(doindex,dovalue,color='green') 
plt.show()

length=5
part=[]
last=0
total=[]
for i in range(thetaback,len(bccc)):
    if i in upindex:
        part.append(i)
        last=i
    elif i-last<length:
        part.append(i)
    elif i-last==length:
         total.append(part)
         part=[]
#plt.plot(bccc)  
#cul=0    
#win=0  
#for i in range(len(total)):
#    plt.plot(num[total[i][0]:total[i][-1]],bccc[total[i][0]:total[i][-1]],color='red') 
##    print(total[i][0],total[i][-1],bccc[total[i][0]]-bccc[total[i][-1]]*1.001)
#    cul=cul+bccc[total[i][0]]-bccc[total[i][-1]]*1.001
#    if bccc[total[i][0]]-bccc[total[i][-1]]*1.001>=0:
#        win=win+1
#plt.show()
#print(cul,win/len(total))

sta=-1
buy=[]
sell=[]
plt.plot(bccc)
for i in range(thetaback,len(bccc)):
    if sta==-1 and i in upindex:
        buy.append(i)
        sta=1
    if sta==1 and i in doindex:
        sell.append(i)
        sta=-1
cul1=0
win1=0
for i in range(len(buy)-1):
#    plt.plot(bccc)
    print(i,buy[i],sell[i],bccc[sell[i]]-bccc[buy[i]])
    plt.plot(num[buy[i]:sell[i]],bccc[buy[i]:sell[i]],color='red')
    cul1=cul1+bccc[sell[i]]-bccc[buy[i]]*1.001
    if bccc[sell[i]]>=bccc[buy[i]]*1.001:
        win1=win1+1
#    plt.show()
print(cul1,win1/len(buy))
#    plt.show()
    