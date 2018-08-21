# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:47:58 2018

@author: pxr13258
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

maa=6#  important parameter!!!1
def dm(high,low):
    plusdm=[0]
    minusdm=[0]
    for i in range(1,len(high)):
        if (high[i]-high[i-1])>=(low[i-1]-low[i]) and (high[i]-high[i-1])>=0:
                plusdm.append(high[i]-high[i-1])
        else:
                plusdm.append(0)
    
        if (high[i]-high[i-1])<(low[i-1]-low[i]) and (low[i-1]-low[i])>=0:
            
                minusdm.append(low[i-1]-low[i])
        else:
                minusdm.append(0)
       
    return plusdm,minusdm
def truerange(close,high,low):
    tr=[]
    tr.append(high[0]-low[0])
    for i in range(1,len(high)):
        A=high[i]-low[i]
        B=abs(high[i]-close[i-1])
        C=abs(low[i]-close[i-1])
        tr.append(max([A,B,C]))
    tr=pd.ewma(np.array(tr),maa)
    return tr
def di(dmp,dmm,tr):
    dip=[]
    dim=[]
    dx=[]
    dmp=pd.ewma(np.array(dmp),maa)
    dmm=pd.ewma(np.array(dmm),maa)
    for i in range(len(tr)):
        #print(tr[i])
        dip.append(dmp[i]/tr[i]*100)
        dim.append(dmm[i]/tr[i]*100)
        if dmp[i]+dmm[i]==0:
            dx.append(0)
        else:
           dx.append((abs(dmp[i]-dmm[i])/(dmp[i]+dmm[i]))*100)
    adx=pd.ewma(np.array(dx),maa)      
    return dip,dim,dx,adx
def findtheta(shuzu):
   #print(shuzu.type)
    mi=float(min(shuzu))
    ma=float(max(shuzu))
    xrey=np.arange(mi,ma,float((ma-mi)/len(shuzu)))
    if len(xrey)!=len(shuzu):
         xrey=list(xrey)
         del xrey[-1]
         xrey=np.array(xrey)
 
    [A,B]= np.polyfit(xrey, shuzu, 1)
    
    ang=math.atan(A)
    target=float(ang/1.6)
    return target


import statsmodels.api as sm # 最小二乘
from statsmodels.stats.outliers_influence import summary_table 
def makeols(x,y):
    X=sm.add_constant(x)
    regs=sm.OLS(y,X)
    res=regs.fit()
    para=res.params
    result=res.rsquared
    return float(para[0]),float(para[1]),float(result)

def ols(window,bccc,bccl,bcch):
    beta=[]
    r=[]
    for i in range(window,len(bccc)):
        xx=bccl[i-window:i]
        yy=bcch[i-window:i]
        beta0,r2=makeols(xx,yy)
        beta.append(beta0)
        r.append(r2)
    beta=np.array(beta)
    meanbeta=pd.rolling_mean(beta,window)
    sd_bccc=pd.rolling_std(beta, window)
    re=[]
    re1=[]
    for i in range(window,len(beta)-1):
        rsrs=(beta[i]-meanbeta[i])/sd_bccc[i]
        re.append(rsrs)
        rightdev=rsrs*beta[i]*r[i]
        re1.append(rightdev)
    return beta,re,re1
  
def get_CCI(high,low,close,N):  
  
     
    c=[(high[i]+low[i]+close[i])/3 for i in range(len(close))] 
    typ=np.array(c)
    typ_mean=pd.rolling_mean(typ,N)
    md=pd.rolling_mean(abs(typ-typ_mean),N)
    #typ=typ.reshape(1,len(typ))

    #md=abs(typ-typ.rolling(N).mean())
    cci=(typ-typ_mean)/(0.015*md) 
    return cci  
ethcci=get_CCI(ethh,ethl,ethc,24) 
cci_upstop=[]
for i in range(1,len(ethcci)):
    if ethcci[i]<150 and ethcci[i-1]>150:
        cci_upstop.append([i,ethc[i]])
cci_upstop=np.array(cci_upstop)  

#re the rsrs before fix
#re1 the rsrs after fix