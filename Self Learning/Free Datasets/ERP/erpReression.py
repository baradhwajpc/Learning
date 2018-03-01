# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 06:57:01 2018

@author: baradhwaj
"""

import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import statsmodels.formula.api as smf

os.chdir('G:\Python\Learning\Self Learning\Free Datasets\ERP')

erpData = pd.read_csv('machine.txt',header=None,names=['vendor name',
'Model Name',
'MYCT',
'MMIN',
'MMAX',
'CACH',
'CHMIN',
'CHMAX',
'PRP',
'ERP'])

# DV:ERP-Estimated relative performance, 
# PRP - predicted relative performance
# The data set originally had a column erp (estimated relative performance) estimated by the article
# The published relaative performace , prp is the value which is the actual value calculated at the production
# We can remove the  erp column , and estimate the published relative performance

erpData = erpData.drop('ERP', axis=1)

f,ax = plt.subplots(figsize=(8, 6))
sns.heatmap(erpData.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# MYCT has weak correlation against all variables
# MMMin MMMAX have positive correlation
#  ERP has string positive correlation with MMIN(0.8), MMAX(0.9),
# CACH,CHMIN CHMAX - 0.6 WEAK POSITIVE CORRELATION
# ERP and PRP have a perfect correlation

# Few plots to estimate the relation 
f,ax = plt.subplots(figsize=(4, 3))
sns.pairplot(erpData)

ax = sns.regplot(x="MMAX", y="PRP", data=erpData,fit_reg = False )
ax = sns.regplot(x="MMIN", y="PRP", data=erpData,fit_reg = False )

# We wil split the data to train test

all_rows = np.arange(erpData.shape[0])
# Training - 70 % and test 30 % of the 800 records
trainingRecordCount = round(0.7 * erpData.shape[0]) # 146 for training
testingRecordCount  = round(0.3 * erpData.shape[0]) # 63 for test
np.random.seed(100)
training_samples = np.random.choice(all_rows,trainingRecordCount,replace= False) # row index of data to be used for training
testing_samples = all_rows[~ np.in1d(all_rows ,trainingRecordCount)] # row index of data to be used for testing

# Using iloc get training and test data
erpData_training = erpData.iloc[training_samples,:]
erpData_testing = erpData.iloc[testing_samples,:]

erp_linear_model = smf.ols('PRP ~ MMAX', data = erpData_training).fit()
erp_linear_model.summary()
# P value is insignificant , accepting null hypotysis
# R squared : 0.750  positive 
#-35.4054 +0.0125MMAX

erp_linear_model = smf.ols('PRP ~ MMIN', data = erpData_training).fit()
erp_linear_model.summary()
# P value is significant , rejecting null hypotysis
# R squared : 0.674 - Weak Positive 
# 8.3915 + 0.0337 * MMIN

erp_linear_model = smf.ols('PRP ~ CACH', data = erpData_training).fit()
erp_linear_model.summary()

# Multi Linear Model
erp_multilinear_model = smf.ols('PRP ~ MMIN+MMAX', data = erpData_training).fit()
erp_multilinear_model.summary()
# Adj R Squared : 0786
# -30.7823 +0.0140 * MMIN +0.084 * MMAX
erp_multilinear_model = smf.ols('PRP ~ MYCT+MMIN +MMAX +CHMIN+CHMAX', data = erpData_training).fit()
erp_multilinear_model.summary()
# Adj R ^2 : 0.864
# MYCT and CHMIN - Null hypothisis is significant
# -56.1030 +0.0392 *MYCT +0.0171 * MMIN +0.055  MMAX  +2.4233 * CHMAX

erpData_testing2 = erpData_testing.copy()
del erpData_testing2["PRP"]
predicted_prp = erp_multilinear_model.predict(erpData_testing2)

def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.median(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape
    
# Testing

MAPE(erpData_testing["PRP"],predicted_prp)
# Mean APE = 81.73
# Median APE = 50.85
# Visualizing the model fit

prp_training_as_test = erpData_training.copy()
del prp_training_as_test["PRP"]
prp_training_as_test["predicted_prp_tr"] = erp_multilinear_model.predict(prp_training_as_test)

plt.scatter("MMIN","PRP",data = erpData_training)
plt.scatter("MMIN","predicted_prp_tr",data = prp_training_as_test,c="red")

plt.scatter("MMAX","PRP",data = erpData_training)
plt.scatter("MMAX","predicted_prp_tr",data = prp_training_as_test,c="red")

plt.scatter("MYCT","PRP",data = erpData_training)
plt.scatter("MYCT","predicted_prp_tr",data = prp_training_as_test,c="red")

plt.scatter("CHMIN","PRP",data = erpData_training)
plt.scatter("CHMIN","predicted_prp_tr",data = prp_training_as_test,c="red")

plt.scatter("CHMAX","PRP",data = erpData_training)
plt.scatter("CHMAX","predicted_prp_tr",data = prp_training_as_test,c="red")
