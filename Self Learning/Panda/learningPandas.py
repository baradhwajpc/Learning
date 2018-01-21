# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:22:13 2017

@author: baradhwaj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
s = pd.Series([1,3,4,5,nan,7,nan,9])
print(s)
dates = pd.date_range(20171026,periods=6)
df1 = pd.DataFrame(np.random.rand(6,4),index = dates , columns=list('ABCD'))
print(df1)
df2 = pd.DataFrame({
                    'A': 1,
                    'B': pd.Timestamp('20171025'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(['a','b','c','d']),
                    'F' : 'bar'
                    })
# viewing data
print(df2)
print(df2.dtypes)
print(df2.head())
print(df2.tail(2))
print(df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe)
'df.T
print(df2.sort_index(axis=0,ascending=False))
print(df2.sort_values(By='C'))

# Selection
print(df2['A'])
print(df2[0:3])
print(df2.loc[dates[0]])