# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:58:43 2018

@author: pxr13258
"""

from pyhht.visualization import plot_imfs
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
pi=3.14
t = np.linspace(0, 1, 1000)
modes = np.sin(2 * pi * 5 * t) + np.sin(2 * pi * 10 * t)
x = modes + t
decomposer = EMD(x)
imfs = decomposer.decompose()
plot_imfs(x, imfs, t) 


pathbcc='D:/newdata/BCCBTC_5T.csv'
bcc= pd.read_csv(pathbcc)
data=bcc['Close'].values
emdtest=[]
for i in range(10):
    emdtest.append(data[i*288:(i+1)*288])
emdtest=np.array(emdtest)
dec=EMD(emdtest[9])
imf=dec.decompose()
plot_imfs(emdtest[9], imf) 