# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:22:44 2017
@author: baradhwaj
"""
### Home work ##########
##Dv : wg, IDV: metmin
# Clean Data (remove na) - Done
# Divide wg to taining and test
    # find the  number of test and training data
    # Using seed and random.choice get all rows which are training and test data
    # Create Dataframe with dv , idv for training and test data
    # Generate a plot with dv and idv 
# Build linear model on training data
        # Calculate correlation and coeff
# To calculate : R square, P value is significant , linear equation
# Test on test data 
# To calculate : MAPE
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
# Step 1  - Read csv data
os.chdir('G:\Python\Learning\Lectures\Day-6\data')
weightGain = pd.read_csv("wg.csv",thousands =',')
print(weightGain)
# Step2 - Plot and cleanup
plt.scatter("metmin" ,"wg",data=weightGain)
plt.ylabel("Weight Gain (g)")
plt.xlabel("Metmin")
wgDataclean = weightGain.dropna() # simpler way of cleaning data
# Clean data has 256 rows
plt.scatter("metmin" ,"wg",data=wgDataclean)
plt.ylabel("Weight Gain (g)")
plt.xlabel("Metmin")

# Step 2 - Segregate data as test and train in 70 : 30 ratio
#################### find the  number of test and training data##########
# Take index rows and assign to rows a new series 
weightGainAll_rows = np.arange(wgDataclean.shape[0])
# 70 % of 255 - 179 rows for training
numTrainingSampleCount = round(0.7*wgDataclean.shape[0]) # 
# 30 % of 255 - 77 rows for test
numTestSampleCount = round(0.3*wgDataclean.shape[0]) # 
###############  end of find the  number of test and training data #######

####Using seed and random.choice get all rows which are training and test data#####
np.random.seed(100) ### Randomly select values based on index generated using seed
# Random.Choice ->  1d Array with all data , count to be sliced , replace -> False (Ensures no values are repeated) 
wgTraining_sample = np.random.choice(weightGainAll_rows,numTrainingSampleCount,replace=False)
# Remaining data in weightGainAll_rows is fetched using is not in in1d
wgTesting_sample = weightGainAll_rows[~np.in1d(weightGainAll_rows,wgTraining_sample)]
## end of Using seed and random.choice get all rows which are training and test data#####

##########Create Dataframe with dv , idv for training and test data############
## Use iloc to splice out rows based on index in  wgTraining_sample and wgTesting_sample
wg_trainingDF = wgDataclean.iloc[wgTraining_sample,:]
wg_testingDF = wgDataclean.iloc[wgTesting_sample,:]
# Generate plot for training data
plt.scatter("metmin","wg",data=wg_trainingDF)
plt.xlabel("metmin")
plt.ylabel("BodyWeight Gain")
######### end of Create Dataframe with dv , idv for training and test data#####
######### Build linear model on training data #################
###### Calculate correlation and covariance and determine the correlation #####
# -1 : String neg corr
# +1: Strong pos corr
# 0 - No corr  Note: -.5 to 0 to +.5 is the scale to inform no corr can be done
## Proceed with reg wien there is reasonably strong corr ###
wg_trainingDF['metmin'].cov(wg_trainingDF['wg']) # Covariance -> -6055.42
wg_trainingDF['metmin'].corr(wg_trainingDF['wg']) # Correlation -> -0.91285
#Strong Negative Correlation  between weight gain and metmin( not in .5 to -.5)
# Weight Gain and Metmin and indirectly propotional
#### End of Calculate correlation and covariance and determine the correlation ####

### Build Regression Model ### 
# To calculate : R square, P value is significant , linear equation
wg_simple_Linear_TrainingModel = smf.ols(formula = 'wg ~ metmin' , data=wg_trainingDF).fit()
wg_simple_Linear_TrainingModel.summary()
# R Squareed is the standard metric to measure a good fit
# R squared is 0.833 faily ok fit
# R squared should be between 0 and 1 
# 0 is a bad model and 1 is a good model
# p value should be less than .05  should be included
# p value should be less thans .05 for idv to be significant
# Regression dos a hypotisis testing
# H0 ( Null Hypothysis ): No relation between dv and idv
# if p is greater than 0.95 , I accpet null hypothisis with 95 % confidence
# if p is less than 0.95 , I reject null hypothisis with 95 % confidence
# P value - 0.000  -  intercept and coeff is significant
# metmin = m*wg + C
# formula : -0.0186 * metmin + 53.4849

##################End of Build Regression Model##############################

############################## Evaluate with test data ########################
wgTraining_as_TestData = wg_testingDF.iloc[:,0:4]
## Predict metmin for weight gain  in testing data
predicting_wg_testingData = wg_simple_Linear_TrainingModel.predict(wgTraining_as_TestData)
abs_percentage_wgerrorTestData = abs(wg_testingDF['wg'] - predicting_wg_testingData) / wg_testingDF['wg']

wgModel_eval_training = pd.DataFrame({'metmin': wg_testingDF['metmin'],
                           'Actual wg': wg_testingDF['wg'],
                            'Predicted wg' : predicting_wg_testingData,
                            'Abs Percentage Error': abs_percentage_wgerrorTestData
                           })

def MAPE(actual,predicted):
    abs_precent_diff = abs(actual - predicted) / actual
    abs_percent_diff = abs_precent_diff.replace(np.inf,np.nan)  # Avoid infinity error
    median_ape = np.median(abs_percent_diff)
    mean_ape = np.mean(abs_percent_diff)
    mape = pd.Series([mean_ape,median_ape],index = ["Mean_APE","Median_APE"])
    return mape

np.mean(wgModel_eval_training['Abs Percentage Error']) #41.72 # MAPE
np.median(wgModel_eval_training['Abs Percentage Error']) #24.12 # MEPE - Median abs percentage error
MAPE(wg_testingDF['wg'],predicting_wg_testingData)

plt.scatter(wgModel_eval_training['metmin'],wgModel_eval_training['Actual wg'])
plt.scatter(wgModel_eval_training['metmin'],wgModel_eval_training['Predicted wg'],c='red')
plt.xlim([0,5000])
plt.ylim([-60,100])

## The relation is non linear due to improper model fit
# More than 1st order relatonship
# y = mX+ c
# y = aX^2 + bX + C
# y = aX^3 + bX^2+cX+d
# Dont go beyond 2nd order
# Non Linear Releation : a*metmin  + b* metmin^2 + Intercept
wg_non_linear_model = smf.ols("wg~metmin + np.power(metmin,2)", data=wg_trainingDF).fit()
wg_non_linear_model.summary()
# R Squared = 0.977 Intercept : 95.43 
# Pvalues is significant
predicted_wg_nonlin = wg_non_linear_model.predict(wgTraining_as_TestData)

abs_percentage_wgerrornonLinear_TestData = abs(wg_testingDF['wg'] - predicted_wg_nonlin) / wg_testingDF['wg']
                                     
MAPE(wg_testingDF['wg'],predicted_wg_nonlin) # Mean : 17.98 and MEdian : 8.03 %


wgModel_eval_nonlinear_training = pd.DataFrame({'metmin': wg_testingDF['metmin'],
                           'Actual wg': wg_testingDF['wg'],
                            'PredictedNLwg' : predicted_wg_nonlin,
                            'Abs Percentage Error NL': abs_percentage_wgerrornonLinear_TestData
                           })

plt.scatter(wgModel_eval_nonlinear_training['metmin'],wgModel_eval_nonlinear_training['Actual wg'])
plt.scatter(wgModel_eval_nonlinear_training['metmin'],wgModel_eval_nonlinear_training['PredictedNLwg'],c='red')
plt.scatter(wgModel_eval_training['metmin'],wgModel_eval_training['Predicted wg'],c='yellow')
plt.xlim([0,5000])
plt.ylim([-60,100])



###### Data Transformation #####
## Instead of building non linear model on raw data , data can be transformed to a log scale and model can be linear
## Log - Linear Model 
# Linear  - Log
# Log - Log
wgDataclean['log_wg'] = np.log(wgDataclean['wg'])
plt.scatter('metmin','log_wg',data =wgDataclean)

################################################################################## Data Train
##################################################################
# Step 3  correletion : -0.90625914221504655
weightGain["metmin"].corr(weightGain["wg"]) 
# As activity increases weight gain decreases
# Step 4: 
wg_linear_model = smf.ols(formula='wg ~ metmin', data = weightGain).fit()
# ordinary least square method
wg_linear_model.summary()
# Formula: Wg = m* metmin + C
# Substitution : wg =-0.0189*metmin +54.1624
#15 , 	1790
metmin = 1790
predictedwg = -0.0189*metmin +54.1624
print(predictedwg)
actualWg = 15
abs(actualWg -predictedwg )/actualWg # 35.54 % deviation
# how much deviation is permissable / can be used as quantifier to assume the data is correct and proceed?

