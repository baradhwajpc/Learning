# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 07:57:05 2017

@author: baradhwaj
"""
## Data :  This is the data corresponding to AWS (Amazon Web Services) spot
#instance (https://aws.amazon.com/ec2/spot/). Your chance for retaining a spot (availability)
#generally increases when you bid with a higher price

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn
# Read the csv files and print the 
os.getcwd()
os.chdir('G:\Python\Learning\Lectures\Day-10\Linear Regression\Homework\Data')
availablityData = pd.read_csv('Availability.csv')
print(availablityData)

# Step 1 : Identify dv and idv
# Availablity : Dv
# Bid Price : Idv
# Scatter plot : Bid Proce vs Availablity
plt.scatter('Bid','Availability',data=availablityData)
plt.xlabel("Bid Value") # IDV to x 
plt.ylabel("Availablity") # DV to y 

## Step -2 :Find correlation 
# 0 - No corr  Note: -.5 to 0 to +.5 is the scale to inform no corr can be done
availablityData["Bid"].corr(availablityData["Availability"]) # 0.62355 - Decent positive correlation

## Step 3 : Training Test split
all_rows = np.arange(availablityData.shape[0])
# Training - 70 % and test 30 % of the 800 records
trainingRecordCount = round(0.7 * availablityData.shape[0]) # 560 for training
testingRecordCount = round(0.3 * availablityData.shape[0]) # 240 for test
np.random.seed(100)
training_samples = np.random.choice(all_rows,trainingRecordCount,replace= False) # row index of data to be used for training
testing_samples = all_rows[~ np.in1d(all_rows ,training_samples)] # row index of data to be used for testing

# Using iloc get training and test data
availablityData_training = availablityData.iloc[training_samples,:]
availablityData_testing = availablityData.iloc[testing_samples,:]


#3.1 Finding out the Covariance & Correlation
availablityData_training['Bid'].cov(availablityData_training['Availability']) 
#3.2 Covariance is nearly zero (0.00091)                                          
availablityData_training['Bid'].corr(availablityData_training['Availability']) 
# 0.6391 Strongly +ve Correlation

## Step 4 : build Linear regression model
availablityData_training_model = smf.ols(formula = 'Availability ~ Bid',data=availablityData_training).fit()
availablityData_training_model.summary()
# R-squared is the standard metric to measure goodness of fit
# R-squared will be between 0 and 1
# R-squared = 0.408 # bad model fit
# P value is 0 . IDv - Bid is significant
# availablity =  Bid * 51.6434 -0.6559

def MAPE(actual,predicted):
    abs_percent_diff = abs(actual-predicted)/actual
    # 0 actual values might lead to infinite errors
    # replacing infinite with nan
    abs_percent_diff = abs_percent_diff.replace(np.inf,np.nan)
    median_ape = np.median(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape
# Step 5 : Test on test data
# 5.1: Copy test dataset to a new dataset with index, dv and idv columns
availablityData_testingcopy2 = availablityData_testing.copy()
del availablityData_testingcopy2["Availability"]
#5.2: Create dataframe by applying predict method of model on the dataset created in the previous step . 
#    This is predicted testing data for dependent variable
predicted_Availability = availablityData_training_model.predict(
                                                         availablityData_testingcopy2)
#5.3: Calculate absolute error in test data by finding the difference between (actual dependent variable - predicted testing data for dependent variable) / actual dependent variable
#5.4: Calculate the mean value of the previous step . This is mape - Mean Absolute percentage error.
MAPE(availablityData_testing["Availability"],predicted_Availability)
# MAPE : 2.481933
# 5.5 Get  a copy of training data
availability_training_as_test = availablityData_training.copy()
del availability_training_as_test["Availability"]
availability_training_as_test["predicted_Availability"] = availablityData_training_model.predict(availability_training_as_test)
# 5.6: Plot a scatter with both actual dependent variable and predicted dependent variable
plt.scatter('Bid','Availability',data = availablityData_training)
plt.scatter("Bid","predicted_Availability",data = availability_training_as_test,c="red")
 
###################################end ########################################
######### Note: BAd correlation . Should have stopped there. But did for practice
    
# Step 1 : Identify dv and idv
# spot price : Dv
# bid : Idv
# Scatter plot : Spotprice vs Availablity
plt.scatter('Spotprice','Availability',data=availablityData)
plt.xlabel("Spotprice") # IDV to x 
plt.ylabel("Availability") # DV to y 

## Step -2 :Find correlation 
# 0 - No corr  Note: -.5 to 0 to +.5 is the scale to inform no corr can be done
availablityData["Spotprice"].corr(availablityData["Availability"]) 
# -0.17284 - Bad positive correlation - Ideally should not proceed

## Step 3 : Training Test split - Can skip
all_rows = np.arange(availablityData.shape[0])
# Training - 70 % and test 30 % of the 800 records
trainingRecordCountSP = round(0.7 * availablityData.shape[0]) # 560 for training
testingRecordCountSP = round(0.3 * availablityData.shape[0]) # 240 for test
np.random.seed(100)
training_samplesSP = np.random.choice(all_rows,trainingRecordCountSP,replace= False) # row index of data to be used for training
testing_samplesSP = all_rows[~ np.in1d(all_rows ,training_samplesSP)] # row index of data to be used for testing

# Using iloc get training and test data
availablityData_trainingSP = availablityData.iloc[training_samplesSP,:]
availablityData_testingSP = availablityData.iloc[testing_samplesSP,:]


availablityData_trainingSP['Spotprice'].cov(availablityData_trainingSP['Availability']) 
#3.2 Covariance is nearly zero (0.00091)                                          
availablityData_trainingSP['Spotprice'].corr(availablityData_trainingSP['Availability']) 
#-0.174641916 - Weak - correlation  :(
# Step 4: 
availablityData_training_modelSP = smf.ols(formula = 'Availability ~ Spotprice',data=availablityData_trainingSP).fit()
availablityData_training_modelSP.summary()

# R-squared is the standard metric to measure goodness of fit
# R-squared will be between 0 and 1
# R-squared = 0.030  # bad model fit
# P value is 0 . IDv - SpotPrice is significant
# availablity = SpotPrice * (-22.9312) +1.1067 
# Step 5 : Test on test data
# 5.1: Copy test dataset to a new dataset with index, dv and idv columns
availablityData_testingcopy3 = availablityData_testingSP.copy()
del availablityData_testingcopy3["Availability"]
#5.2: Create dataframe by applying predict method of model on the dataset created in the previous step . 
#    This is predicted testing data for dependent variable
predictedSP_Availability = availablityData_training_modelSP.predict(
                                                         availablityData_testingcopy3)
#5.3: Calculate absolute error in test data by finding the difference between (actual dependent variable - predicted testing data for dependent variable) / actual dependent variable
#5.4: Calculate the mean value of the previous step . This is mape - Mean Absolute percentage error.
MAPE(availablityData_testing["Availability"],predictedSP_Availability)
# Invalid value encountered in median
# MAPE : 4.09781
# 5.5 Get  a copy of training data
availability_training_as_testSP = availablityData_training.copy()
del availability_training_as_testSP["Availability"]
availability_training_as_testSP["predicted_Availability"] = availablityData_training_modelSP.predict(availability_training_as_testSP)
# 5.6: Plot a scatter with both actual dependent variable and predicted dependent variable
plt.scatter('Spotprice','Availability',data = availablityData_trainingSP)
plt.scatter("Spotprice","predicted_Availability",data = availability_training_as_testSP,c="red")
 
###########################End#################################################

corr_matrix = availablityData_training.corr()
# Availability vs Bid : 0.620356 
# Availability vs Spotprice: -0.172840

availabilityModelBidSP = smf.ols(formula = 'Availability ~ Bid+Spotprice',data = availablityData_training).fit()
availabilityModelBidSP.summary()
# Adj R ^2 : 0.613
# 0.5350 +65.0852 * Bid -67.2370 *spotprice
# P value is 0 

availablityData_testingcopy4 = availablityData_testing.copy()
del availablityData_testingcopy4["Availability"]
#5.2: Create dataframe by applying predict method of model on the dataset created in the previous step . 
#    This is predicted testing data for dependent variable
predicted_AvailabilityBIDSP = availabilityModelBidSP.predict(
                                                         availablityData_testingcopy4)
MAPE(availablityData_testing["Availability"],predicted_AvailabilityBIDSP)
# Mean : 2.030226
# Median : NAN

availability_training_as_testSPBID = availablityData_training.copy()
del availability_training_as_testSPBID["Availability"]
availability_training_as_testSPBID["predicted_Availability"] = availabilityModelBidSP.predict(availability_training_as_testSPBID)


plt.scatter('Spotprice','Availability',data = availablityData_trainingSP)
plt.scatter("Spotprice","predicted_Availability",data = availability_training_as_testSPBID,c="red")


plt.scatter('Bid','Availability',data = availablityData_trainingSP)
plt.scatter("Bid","predicted_Availability",data = availability_training_as_testSPBID,c="red")
