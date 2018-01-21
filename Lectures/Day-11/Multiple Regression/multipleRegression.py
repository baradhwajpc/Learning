# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:48:09 2017

@author: baradhwaj
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
## Cement grade analysis :  Heat evolved when cement sets,  if exceeds threshold cement will not be approved
# x1,x2,x3 and x4 -> Component  ( 4 key ingredents )ratios y -> time for setting
# Step 0 :  Read data from csv file 
os.getcwd()
os.chdir("G:\Python\Learning\Lectures\ML-Dataset")
cementData = pd.read_csv("cement.csv",thousands =',')
print(cementData)

# Step 1 : Identify dependent and idv 
# DV -> y - Heat evolved
# IDV -> x1,x2,x3,x4s Composition of key ingredents  
# Step -2 : Visualize the data
plt.scatter('x1','y',data=cementData) # x increases y increasess ( noisy data)
plt.scatter('x2','y',data=cementData) # x incr y incre ( better releated)
plt.scatter('x3','y',data=cementData) # x3 increase y decrease ( noisy)
plt.scatter('x4','y',data=cementData) #x4 increase y decrease (better related)

axes = pd.tools.plotting.scatter_matrix(cementData,alpha=0.8)

## Step 3 calculates Correlation 
cementData['x1'].corr(cementData['y']) # 0.73s
cementData['x2'].corr(cementData['y']) # 0.81
cementData['x3'].corr(cementData['y']) # -0.53
cementData['x4'].corr(cementData['y']) # -0.82s

# or build correlation matrix
corr_matrix = cementData.corr()

# Step 4 : Build Model
cementModel = smf.ols(formula = 'y ~ x1+x2+x3+x4',data = cementData).fit()
cementModel.summary()
# R Squared - 0.982 Good value close to 1  More variable increases
# Adj R Squared : 0.974  # if the new ind var adds value to model the new is added

#y = 1.5511 * x1 + 0.5102 * x2 + 0.1019 * x3 - 0.1441 * x4 + 62.4054
# Intercept  62.4054
# P value : 0.399 , 0.071 , 0.501 , 0.896 , 0.8441 not significant
# p value should be less than .05  should be included
# p value should be less thans .05 for idv to be significant
# All p values are significant

## B uild Models - Look R Sqlared and p Values 
# x1
cementModelX1 = smf.ols(formula = 'y ~ x1',data = cementData).fit()
cementModelX1.summary() # R Squared 0.534 Intercet: 81.4793 and 1.8687 p value : 0.005
# X2
cementModelX2 = smf.ols(formula = 'y ~ x2',data = cementData).fit()
cementModelX2.summary() # R Squared : 0.666 Intercept : 57.4237 , 0.7591 p : 0.001
# x3
cementModelX3 = smf.ols(formula = 'y ~ x3',data = cementData).fit()
cementModelX3.summary() # R squared : 0.286 Intercept : 110.2027 , -1.2558 p : 0.060
# x4
cementModelX4 = smf.ols(formula = 'y ~ x4',data = cementData).fit()
cementModelX4.summary() # R squared : 0.675 , Intercept : 117.5679  , -0.7382 , p : 0.001
# x1,x2
cementModelX1X2 = smf.ols(formula = 'y ~ x1+x2',data = cementData).fit()
cementModelX1X2.summary() # adj R squared : 0.974  , Intercept : 52.5773 ,1.4683 , 0.6623 , p : all 0
# x1,x3
cementModelX1X3 = smf.ols(formula = 'y ~ x1+x3',data = cementData).fit()
cementModelX1X3.summary() # adj r squared : 0.458 sIntercept : 72.3490s 2.3125 , 0.4945 p values : 0.002 , 0.037 , 0.587
# x1,x4
cementModelX1X4 = smf.ols(formula = 'y ~ x1+x4',data = cementData).fit()
cementModelX1X4.summary() # Adj R squared : 0.972 Intercept : 103.0974 , 1.4400, -0.6140 p : all 0 
# x2,x3
cementModel = smf.ols(formula = 'y ~ x1+x2+x3+x4',data = cementData).fit()
cementModel.summary() # adj R ^ 2 : 0.974 Intercept : 62.4054 x1 : 105511 x2 : 0.5102 x3: 0.1019 x: -0.1441
# P value : 0.399 0.071 0.501 0.896 0.844
# x2,x4
cementModel = smf.ols(formula = 'y ~ x1+x2+x3+x4',data = cementData).fit()
cementModel.summary()
# R squared - Adj : 0.974 Intercept: 62.4054 x1: 1.5511 x2: 0.5102 x3 : 0.1019 x4 : -0.1441
# P : 0.399 0.071 0.501 0.896 0.844
