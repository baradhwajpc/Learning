# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 06:35:07 2017

@author: baradhwaj
"""

import random as rn
import numpy as np
import pandas as pd

npA = np.array(rn.sample(range(10,100), 25))
npB = np.array([5]*25)
npC = np.array(list(range(25,0,-1)))
npD = np.array(list(range(0,50,2)))

random_dict = {
                'A':npA,
                'B':npB,
                'C':npC,
                'D':npD
               }
random_df = pd.DataFrame(random_dict)
random_df.index = list(range(1001,1026,1))
# print col A 
s = pd.Series(random_df.loc[:,"A"])
## 2.
df2 =random_df.loc[:,["A","C"]]
## 3.
df3 =random_df.iloc[:,[0,2]]

# 4. 0 to 5 in s 

print(s)
d = s[0:6] # slice 0th to 5th row
df4 = random_df.iloc[2:19,:] # slice 0th to 3rd row and all columns
## 5 :
colAMedian = np.median(npA)
condn2 = random_df["A"] > colAMedian 
df5 = random_df.loc[condn2]
print(df5)