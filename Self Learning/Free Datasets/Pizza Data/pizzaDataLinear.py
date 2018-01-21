# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:46:26 2017

@author: baradhwaj
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
# Step 1 : 
os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Pizza Data')

pizzaData = pd.read_csv('pizzaData.csv')
print(pizzaData)

# Step 2: scatter plot
# Identify Dp and idv
#X = annual franchise fee ($1000)
#Y = start up cost ($1000)
# find the startup DV for annual  franchaise fee (IDV)
plt.scatter('AnnualFee','StartUpCost',data=pizzaData)

# Step 3 : Find the correleation 
print(pizzaData['AnnualFee'].corr(pizzaData['StartUpCost'])) # Weak positive correlation

# Step 4 : generate model
pizzaDataModel = smf.ols('StartUpCost ~ AnnualFee',data=pizzaData).fit()
pizzaDataModel.summary()
# Formula: 867.6042 + 0.3732 * Annual fee
# R Squared is low at 0.228  - Bad model fit


def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.median(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape

pizzaDataTesting = pizzaData.copy()
del pizzaDataTesting["StartUpCost"]
#5.2: Create dataframe by applying predict method of model on the dataset created in the previous step . 
#    This is predicted testing data for dependent variable
predicted_startUp = pizzaDataModel.predict(pizzaDataTesting)
#5.3: Calculate absolute error in test data by finding the difference between (actual dependent variable - predicted testing data for dependent variable) / actual dependent variable
#5.4: Calculate the mean value of the previous step . This is mape - Mean Absolute percentage error.
MAPE(pizzaData["StartUpCost"],predicted_startUp)
pizzaData['PredictedStartUp'] =  predicted_startUp 0.05 , 0.038998

# Mean APE : 0.053

plt.scatter('AnnualFee','StartUpCost',data = pizzaData)
plt.scatter("AnnualFee","PredictedStartUp",data = pizzaData,c="red")
 
########## Non Linearr

pizza_non_linear_model = smf.ols('StartUpCost ~ AnnualFee + np.power(AnnualFee,2)', 
                              data = pizzaData).fit()
pizza_non_linear_model.summary()
# 0.530 Decent R value. All P significant

pizzaDataTestingNonLinear = pizzaData.copy()
del pizzaDataTestingNonLinear["StartUpCost"]
#5.2: Create dataframe by applying predict method of model on the dataset created in the previous step . 
#    This is predicted testing data for dependent variable
predicted_nonLinear_startUp = pizza_non_linear_model.predict(pizzaDataTestingNonLinear)
#5.3: Calculate absolute error in test data by finding the difference between (actual dependent variable - predicted testing data for dependent variable) / actual dependent variable
#5.4: Calculate the mean value of the previous step . This is mape - Mean Absolute percentage error.
MAPE(pizzaData["StartUpCost"],predicted_nonLinear_startUp)
pizzaDataTestingNonLinear['PredictedNLStartUp'] =  predicted_nonLinear_startUp
# 0.0456 , 0.037149

plt.scatter('AnnualFee','StartUpCost',data = pizzaData)
plt.scatter("AnnualFee","PredictedNLStartUp",data = pizzaDataTestingNonLinear,c="red")
