# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 08:12:50 2018

@author: baradhwaj
"""

#1.1

import pandas as pd
import numpy as np

heights_A = pd.Series(data = [176.2, 158.4, 167.6, 156.2,161.4],index = ['s1', 's2', 's3', 's4','s5'])
print(heights_A.shape)
#1.2
weights_A = pd.Series(data = [85.1, 90.2, 76.8, 80.4,78.9],index = ['s1', 's2', 's3', 's4','s5'])
print(weights_A.dtype)
#1.3
df_A = pd.DataFrame({'Student_height':heights_A, 'Student_weight': weights_A},index = ['s1', 's2', 's3', 's4','s5'])
#1.4
heights_B  = pd.Series(np.random.normal(170,25,5),index = ['s1', 's2', 's3', 's4','s5'])
weights_B  = pd.Series(np.random.normal(75,12,5),index = ['s1', 's2', 's3', 's4','s5'])

#1.5
df_B = pd.DataFrame({'Student_height':heights_B, 'Student_weight': weights_B},index = ['s1', 's2', 's3', 's4','s5'])
print(df_B.columns)
