# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 20:36:22 2017

@author: baradhwaj
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
# Prep Step
os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Machine Learning\data')
adData = pd.read_csv('Advertising.csv')
print(adData)

# Step 1:Identify Dep and IDV 
# Dependent : sales
# Independent : TV , radio , newspaper
# considering TV to build single leniar model

#  Step2:Visualize DV and IDV
# Graph looks likne non linear , but we try to see if linear fits in
plt.scatter('TV','sales', data = adData)

# Step 3 : Do a correlation analysis
# Correlation is decent : positive # 0.78222
adData['TV'].corr(adData['sales']) # 0.78222

#Split data for train and test
# Method for train , test segregation
def trainTestSegregation(df,seed_id) : 
    all_records= np.arange(df.shape[0])
    trainingRecordCount = round(0.7 *df.shape[0])
    testRecordCount = round(0.3 * df.shape[0])
    np.random.seed(seed_id)
    trainingRecordsIds = np.random.choice(all_records,trainingRecordCount ,replace=False)
    testingRecordsIds =all_records[~np.in1d(all_records,trainingRecordsIds)] 
    trainingRecords = df.iloc[trainingRecordsIds,:]
    testRecords = df.iloc[testingRecordsIds,:]
    return trainingRecords, testRecords
# Call method with seed 100 to generate train test data
adDataTrain , adDataTest = trainTestSegregation(adData,100)


# Sep 4: Build model  - Look for R ^2 and p value
# R2 range will be between 0 and 1;  close to 1 is agood model
# R squared value : 0.513  - decent model
# P is significant for idv and intercept
# Both conditions are satisfied , go to next step
ad_train_model = smf.ols(formula = 'sales ~ TV',data= adDataTrain).fit()
ad_train_model.summary()

def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.nanmedian(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape
    
mean_ape_all = []
median_ape_all = []
# Step 5: Evaluate model fitness
def predeictDepVariable(df,col_name,model): 
    test_data2 = df.copy()
    del test_data2[col_name]
    # Predicting Availability using Built model
    df["predicted_lin"] = model.predict(test_data2)
    # Evaluating the model
    mape = MAPE(df[col_name],df["predicted_lin"])
    mean_ape_all.append(mape["Mean_APE"])
    median_ape_all.append(mape["Median_APE"]) 

predeictDepVariable(adDataTest,'sales',ad_train_model)

# Mean : 27.19
# Median : 18.20
