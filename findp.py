# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:18:10 2018

@author: pxr13258
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from functions import makeols
pathltc='D:/newdata/LTCUSDT_30T.csv'
ltc= pd.read_csv(pathltc)
pathbtc='D:/newdata/BTCUSDT_30T.csv'
btc= pd.read_csv(pathbtc)
pathbcc='D:/newdata/BCCUSDT_30T.csv'
bcc= pd.read_csv(pathbcc)
patheth='D:/newdata/ETHUSDT_30T.csv'
eth= pd.read_csv(patheth)
ltcc=ltc['Close'].values
#ltcc=ltcc[460:960]
btcc=btc['Close'].values
#btcc=btcc[460:960]
bccc=bcc['Close'].values#[0:-2]
#bccc=bccc[460:960]
ethc=eth['Close'].values[0:-1]
#ethc=ethc[460:960]
#coin=pd.DataFrame({'ltc':ltcc,'btc':btcc,'bcc':bccc,'eth':ethc})
coin=pd.DataFrame({'bcc':bccc,'eth':ethc})
def find_cointegrated_pairs(dataframe):
    n = dataframe.shape[1]
    pvalue_matrix = np.ones((n, n))
    keys = dataframe.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            stock1 = dataframe[keys[i]]
            stock2 = dataframe[keys[j]]
            result = sm.tsa.stattools.coint(stock1, stock2)
            pvalue = result[1]
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
               pairs.append((keys[i], keys[j]))
    return pvalue_matrix, pairs

coin_list = ["bcc", "eth"]
coin_value={"bcc":bccc, "eth":ethc}
#coin_rever={bccc:"bcc", btcc:"btc", ethc:"eth", ltcc:"ltc"}
pvalues, pairs = find_cointegrated_pairs(coin) 
pair=[]

            
#sns.heatmap(1-pvalues, xticklabels=coin_list, yticklabels=coin_list, cmap='RdYlGn_r', mask = (pvalues == 1))
#print(pairs)

def adftest(price):
    seriess = pd.Series(price) # 获取data过程省略
    diff1 = seriess.diff(1)
    diff1[0]=np.mean(diff1)# dta[0] is nan
    result0=sm.tsa.stattools.adfuller(seriess)
    result1=sm.tsa.stattools.adfuller(diff1)
    return result0,result1
def qualified_series(result,lag):
    bar=result[4][lag]
    if result[0]<bar and result[1]<0.05:
        return True
    else:
        return False

def residual_test(b,k,x,y,lag):
    residual=y-k*x-b
    res1,res2=adftest(residual)
    judgement=qualified_series(res1,lag)
    return judgement
def pair_trading_prepare(coinframe,coinlist,coinvalue):
    pvalues, pairs = find_cointegrated_pairs(coinframe) 
    if pairs==[]:
        return False
    pair=[]
    
    for items in coin_list:
        for item in coin_list:
            if (items,item) in pairs:
               pair.append({items:coin_value[items],item:coin_value[item]})
    for coinpair in pair:
        coinpairs=[]
        for p,v in coinpair.items():
            coinpairs.append(v)
        re1,re2=adftest(coinpairs[0])
        re3,re4=adftest(coinpairs[1])
        judge1=qualified_series(re1,'1%')
        judge2=qualified_series(re2,'1%')
        judge3=qualified_series(re3,'1%')
        judge4=qualified_series(re4,'1%')
        trading_pair=[]
   
        if judge1==False and judge2==True and judge3==judge1 and judge4==judge2:
            x=coinpairs[0]
            y=coinpairs[1]
            b1,k1,r=makeols(x,y)
            if residual_test(b1,k1,x,y,'1%')==True:
                trading_pair.append(coinpair)
    
    if trading_pair==[]:
        return False
    else:
        return trading_pair[0]
 
def pair_trading(trade_pair,bar1,bar2,bar3,bar4,slope):
     x=trade_pair[0]
     y=trade_pair[1]
   
     delta=y-slope*x
     xamount=[0]
     yamount=[0]
     sta=-1
     lastsignal=None#
     ha1=[]
     ha2=[]
     ha3=[]
     ha4=[]
     k=slope
     for i in range(1,len(delta)):
         if sta==-1:
             if delta[i]>bar1 and delta[i-1]<bar1 and lastsignal==None:
                 xamount.append(k)
                 yamount.append(-1)
                 sta=1
                 lastsignal='more'
                 ha1.append(i)
             elif delta[i]<bar2 and delta[i-1]>bar2 and lastsignal==None:
                 xamount.append(-k)
                 yamount.append(1)
                 sta=1
                 lastsignal='less'
                 ha2.append(i)
             else:
               xamount.append(0)
               yamount.append(0) 
         elif sta==1:
            if delta[i]<bar3 and delta[i-1]>bar3 and lastsignal=='more':
                xamount.append(-k)
                yamount.append(1)
                sta=-1
                lastsignal=None
                ha3.append(i)
            elif delta[i]>bar4 and delta[i-1]<bar4 and lastsignal=='less':
                 xamount.append(k)
                 yamount.append(-1)
                 ha4.append(i)
                 sta=-1
                 lastsignal=None
            else:
               xamount.append(0)
               yamount.append(0) 
         else:
               xamount.append(0)
               yamount.append(0)
#     print(ha1,ha3,ha2,ha4)
     return xamount,yamount

    

def cashflow(trade_pair,xamount,yamount):
      x=trade_pair[0]
      y=trade_pair[1]
      cash=0
      tolist=[]
      cashlist=[]
      for i in range(len(x)):
          cashlist.append(cash)
          positionx=sum(xamount[0:i+1])
          positiony=sum(yamount[0:i+1])
          if xamount[i]!=0 and yamount[i]!=0:
             cash=cash-xamount[i]*x[i]-yamount[i]*y[i]#-0.0005(abs(xamount[i]*x[i])+abs(yamount[i]*y[i]))
          total=cash+positionx*x[i]+positiony*y[i]
          tolist.append(total)
      return tolist,cashlist
def cashcal(close,xamount):
    cash=0
    tolist=[]
    cashlist=[]
    po=[]
    for i in range(len(close)):
          cashlist.append(cash)
          positionx=sum(xamount[0:i+1])
         
          if xamount[i]!=0:
             cash=cash-xamount[i]*close[i]#-0.0005(abs(xamount[i]*x[i])+abs(yamount[i]*y[i]))
          total=cash+positionx*close[i]
          tolist.append(total)
          po.append(positionx)
    return tolist,cashlist,po
#检测到协整信号后，在测试区间找到最好的bar1,bar2,bar3,bar4,在接下来的1/5测试天内·滚动测试

barlist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]#0.5,0.6,0.7,0.8,0.9,
def findbar(trade_pair,barlist,i):
    
    x=trade_pair[0]
    y=trade_pair[1]
    b,k,r=makeols(x,y)
    delta=y-k*x
    meann=np.mean(delta)
    stdd=np.std(delta)
    money=[]
    for kk in barlist:
        bar1=float(meann+kk*stdd)
        bar2=meann-kk*stdd
        bar3=meann
        bar4=meann
        xamo,yamo=pair_trading(trade_pair,bar1,bar2,bar3,bar4,k)
        total,cash=cashflow(trade_pair,xamo,yamo)
        money.append(total[-1])  
    maxmoney=money.index(max(money))
    if i==132 :
        print(i,money)
#    plt.plot(money)
    return meann+(maxmoney+1)/10*stdd,meann-(maxmoney+1)/10*stdd,meann,meann,max(money),(maxmoney+1)/10


def buysignal(delta0,delta1,sta,lastsignal,xamount,yamount,bar1,bar2,bar3,bar4,k):
     if sta==-1:
#             print(delta0,bar1)
             if delta0>bar1 and delta1<bar1 and lastsignal==None:
                 xamount.append(k)
                 yamount.append(-1)
                 sta=1
                 lastsignal='more'
                
             elif delta0<bar2 and delta1>bar2 and lastsignal==None:
                 xamount.append(-k)
                 yamount.append(1)
                 sta=1
                 lastsignal='less'
                 
             else:
               xamount.append(0)
               yamount.append(0) 
     elif sta==1:
            if delta0<bar3 and delta1>bar3 and lastsignal=='more':
                xamount.append(-k)
                yamount.append(1)
                sta=-1
                lastsignal=None
                
            elif delta0>bar4 and delta1<bar4 and lastsignal=='less':
                 xamount.append(k)
                 yamount.append(-1)
                
                 sta=-1
                 lastsignal=None
            else:
               xamount.append(0)
               yamount.append(0) 
     else:
               xamount.append(0)
               yamount.append(0)
     return sta,lastsignal,xamount,yamount
lastmove=0
#long=300#300/////ETH VS BcC,LTC
#eth bcc 60, eth ltc 60, bcc ltc 400,btc ltc100/60
#short=80
long=10#60
cul=0
cul1=[]
xamo=[]
yamo=[]
lastk=0
lastsignal=None
sta=-1
delta1=0
dellist=[]
klist=[]
dlist=[]
barli=[]
stalist=[]
cashlist=[]
kklist=[]
lastb,lastk,lastr=makeols(bccc[0:long],ethc[0:long])
for i in range(long):
    xamo.append(0)
    yamo.append(0)
    dellist.append(0)
    dlist.append(0)
    barli.append(0)
    stalist.append(0)
    cashlist.append(0)
    kklist.append(0)
#for i in range(long,len(ethc)):
#  
#    coin=pd.DataFrame({'bcc':bccc[i-long:i],'eth':ethc[i-long:i]})
#    coin_list = ["bcc", "eth"]
#    coin_value={"bcc":bccc[i-long:i], "eth":ethc[i-long:i]}
#    targetpair=pair_trading_prepare(coin,coin_list,coin_value) 
##    if targetpair!=False:
##        print(i-100,i)
#   
#    if targetpair!=False:
#        
#        
#        tp=[bccc[i-long:i],ethc[i-long:i]]
#        b,k,r=makeols(tp[0],tp[1])
#        lastk=k
#        delta=ethc[i]-lastk*bccc[i]
#       
#        delta1=ethc[i-1]-lastk*bccc[i-1]
#        bar1,bar2,bar3,bar4,money,kk=findbar(tp,barlist)
#      
#        sta,lastsignal,xamo,yamo=buysignal(delta,delta1,sta,lastsignal,xamo,yamo,bar1,bar2,bar3,bar4,lastk)  
##        print(i,lastsignal,lastk,delta,delta1)
#        lastmove=i
#    elif i<lastmove+short and targetpair==False:
#        delta=ethc[i]-lastk*bccc[i]
#        delta1=ethc[i-1]-lastk*bccc[i-1]
#        sta,lastsignal,xamo,yamo=buysignal(delta,delta1,sta,lastsignal,xamo,yamo,bar1,bar2,bar3,bar4,lastk)
##        print(i,lastsignal,lastk,delta,delta1)
#    elif i-lastmove==short and targetpair==False:
#         delta=ethc[i]-lastk*bccc[i]
##         print(delta)
#         xcanwei=sum(xamo)
#         xamo.append(-xcanwei)
#         ycanwei=sum(yamo)
#         yamo.append(-ycanwei)
#         sta=-1
#         lastsignal=None
#         lastmove=0
#    else:
#        delta=ethc[i]-lastk*bccc[i]
##        print(delta)
#        xamo.append(0)
#        yamo.append(0)
#    dellist.append(delta)
#    klist.append(lastk)
#trade_pair=[bccc,ethc]
#totot=cashflow(trade_pair,xamo,yamo)



for i in range(long,len(ethc)):
  
    coin=pd.DataFrame({'bcc':bccc[i-long:i],'eth':ethc[i-long:i]})
    coin_list = ["bcc", "eth"]
    coin_value={"bcc":bccc[i-long:i], "eth":ethc[i-long:i]}
#    targetpair=pair_trading_prepare(coin,coin_list,coin_value) 
#    if targetpair!=False:
#        print(i-100,i)
    
    tp=[bccc[i-long:i],ethc[i-long:i]]
    b,k,r=makeols(tp[0],tp[1])
    lastk=k
    delta=ethc[i]-lastk*bccc[i]
    dlist.append(lastk)
    delta1=ethc[i-1]-lastk*bccc[i-1]
    dellist.append([delta,delta1])
    bar1,bar2,bar3,bar4,money,kk=findbar(tp,barlist,i)
    sta,lastsignal,xamo,yamo=buysignal(delta,delta1,sta,lastsignal,xamo,yamo,bar1,bar2,bar3,bar4,lastk) 
    kklist.append(kk)
#    if sta==1:
#        print(delta,delta1,bar1,i)
#        print(1)
    stalist.append(sta)
    barli.append([bar1,bar2,bar3,bar4])
trade_pair=[bccc,ethc]
totot,ca=cashflow(trade_pair,xamo,yamo)
bccto,bccca,bccpo=cashcal(bccc,xamo)
ethto,ethca,ethpo=cashcal(ethc,yamo)
plt.plot(totot)
plt.plot(ethc+bccc)
print(totot[-1])
#plt.plot(ethc+0.36*bccc)
#print(residual_test(b1,k1,x,y,'5%'))

#
##plt.plot(ethc); plt.plot(bccc)
##plt.xlabel("Time"); plt.ylabel("Price")
##plt.legend(["eth", "ltc"],loc='best')

##fig, ax = plt.subplots(figsize=(8,6))
##ax.plot(x, y, 'o', label="data")
##ax.plot(x, result.fittedvalues, 'r', label="OLS")
##ax.legend(loc='best')
#
##plt.plot(result.params[1]*ethc-bccc);
##plt.axhline((result.params[1]*ethc-bccc).mean(), color="red", linestyle="--")
##plt.xlabel("Time"); plt.ylabel("Stationary Series")
##plt.legend(["Stationary Series", "Mean"])
#
#def zscore(series):
#    return (series - series.mean()) / np.std(series)
##plt.plot(zscore(result.params[1]*ethc-bccc))
##plt.axhline(zscore(result.params[1]*ethc-bccc).mean(), color="black")
##plt.axhline(1.0, color="red", linestyle="--")
##plt.axhline(-1.0, color="green", linestyle="--")
##plt.legend(["z-score", "mean", "+1", "-1"])
#window=40
#zs=[]
#delta=result.params[1]*ethc-btcc
#for i in range(window):
#    zs.append(0)
#for i in range(window,len(delta)):
#    current=delta[i-window+1:i+1]
#    z=zscore(current)[-1]
#    zs.append(z)
##start 2.55eth 1bcc
#bar1=1.4
#bar11=bar1
#bar2=-1.1
#bar21=bar2
#
##z>bar buy bccc sell 2.55eth buy 1bccc
##z<bar buy 2.55eth sell 1bccc
#sta=-1
#
#actbcc=[0]
#acteth=[0]
#amobcc=[1]
#amoeth=[result.params[1]]
#for i in range(100):
#    actbcc.append(0)
#    acteth.append(0)
#    amobcc.append(1)
#    amoeth.append(result.params[1])
#    
#sta=-1
#k=0
#for i in range(1,len(zs)):
#    last=zs[i-1]
#    if zs[i]>=bar1 and last<bar1:
#        actbcc.append(1)
#        amobcc.append(1)
#        acteth.append(-1)
#        amoeth.append(result.params[1])
#        k=k+1
#    elif zs[i]<bar11 and last>=bar11:
#        actbcc.append(-1)
#        amobcc.append(1)
#        acteth.append(1)
#        amoeth.append(result.params[1])
#        k=k+1
#    elif zs[i]<bar2 and last>=bar2:
#         actbcc.append(-1)
#         amobcc.append(1)
#         acteth.append(1)
#         amoeth.append(result.params[1])
#         k=k+1
#    elif zs[i]>=bar21 and last<bar21:
#        actbcc.append(1)
#        amobcc.append(1)
#        acteth.append(-1)
#        amoeth.append(result.params[1])
#        k=k+1
#    else:
#        actbcc.append(0)
#        amobcc.append(1)
#        acteth.append(0)
#        amoeth.append(result.params[1])
#        
#            
#bccfinal=pd.DataFrame({'action':actbcc,'amount':amobcc})
#df1 = pd.concat( [btc, bccfinal], axis=1) 
#df1.to_csv('btcp.csv',index=False)   
#ethfinal=pd.DataFrame({'action':acteth,'amount':amoeth})
#df2 = pd.concat( [eth, ethfinal], axis=1) 
#df2.to_csv('ethp1.csv',index=False)   

