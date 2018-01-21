# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 20:00:20 2017

@author: baradhwaj
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
# Read csv file
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Challenge Dataset')
pollutionData = pd.read_csv('challenge dataset.txt',header=None)
pollutionData.columns  = ['CO2','Temp']

# Identify Dependent and independent variable
# Dependent Var : Temp
# Independent Var: CO2

# Scatter plot 
plt.scatter('CO2','Temp',data=pollutionData)
# We have a linear relation of the data present here
# Only 97 data is present , will not split the data
# Find the correlation between the CO2 emission and temprature

print(pollutionData['CO2'].corr(pollutionData['Temp'])) 
# A strong posirtive correlation of 0.83783 is found

# Let us build the model now 
pollutionModel = smf.ols('Temp~CO2',data=pollutionData).fit()
pollutionModel.summary()
# To Note : R squared , model formula and P value
# R - Squared: 0.702
# Model : Temp = 1.1930 * CO2 -3.8958
# P value is significant 0 for both intercept and CO2

# Copy a dataset
pollutionDataCopy = pollutionData.copy()
del pollutionDataCopy["Temp"]

def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.median(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape
predictedPollutionData = pollutionModel.predict(pollutionDataCopy)
MAPE(pollutionData['Temp'],predictedPollutionData)

## Mean APE : 128 %  error
## Median APE : 29.61 % 

# DataFrame With both data
allPollutionData = pd.DataFrame({'CO2' : pollutionData['CO2'],
                                 'Actual Temp' : pollutionData['Temp'],
                                 'Predicted Temp' :predictedPollutionData })

# A scatter plot : 
plt.scatter('CO2','Actual Temp',data=allPollutionData)
plt.scatter('CO2','Predicted Temp',data=allPollutionData,color='orange')