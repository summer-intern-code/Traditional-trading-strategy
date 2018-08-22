# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 14:19:54 2018

@author: pxr13258
"""

from WindPy import w
from datetime import *
#.start()
def printpy(outdata):

    if outdata.ErrorCode!=0:

        print('error code:'+str(outdata.ErrorCode)+'\n')
        return()


    for i in range(0,len(outdata.Data[0])):


        strTemp=''
        if len(outdata.Times)>1:
            strTemp=str(outdata.Times[i])+' '
        for k in range(0, len(outdata.Fields)):
            strTemp=strTemp+str(outdata.Data[k][i])+' '
        print(strTemp)



print('\n\n'+'-----通过wsd来提取时间序列数据，比如取开高低收成交量，成交额数据-----'+'\n')
wsddata1=w.wsd("600000.SH", "close,amt", "2018-6-21")
print(wsddata1)

