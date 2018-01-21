# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:30:48 2017

@author: baradhwaj
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import seaborn
# DV: HWT  IDV : BWT
# Step 1 : 
os.getcwd()
os.chdir("G:\Python\Learning\Lectures\Day-10\Data")
catsData = pd.read_csv("cats.csv",thousands =',')
print(catsData)
#Step-2:
plt.scatter("Bwt","Hwt",data=catsData)
plt.xlabel("Body Weight(kg)")
plt.ylabel("Heart Weight(g)")
cat_3kg = catsData.loc[(catsData["Bwt"]>3)&(catsData["Bwt"<= 4])]
np.mean(cat_3kg["Hwt"])
# Hwt = m* Bwt +C
# Step3: Correlation analysis
# Co varianc helps measuing the releationship between variables
# Co var depends on the scale of data
# corrlelation is scaled co variance ranging -1 and 1
# -1 : String neg corr
# +1: Strong pos corr
# 0 - No corr  Note: -.5 to 0 to +.5 is the scale to inform no corr can be done
## Proceed with reg wien there is reasonably strong corr
np.mean(catsData["Bwt"])
np.var(catsData["Bwt"])
np.std(catsData["Bwt"])
catsData["Bwt"].cov(catsData["Hwt"])
catsData["Bwt"].corr(catsData["Hwt"])
#  Step 4 : Build regression model
#formula :DV ~IDV
cats_simple_linear_model = smf.ols(formula='Hwt ~ Bwt', data = catsData).fit()
cats_simple_linear_model.summary()
## Hwt = m* Bwt +C
# formula : 4.0341 * Bwt -0.3567
Bwt = 3.7
predicted_Hwt = 4.0341 * Bwt -0.3567
actual_wt = 11
abs(actual_wt -predicted_Hwt )/actual_wt # 32.4 % deviation

# R Squareed is the standard metric to measure a good fit
# R squared should be between 0 and 1 
# 0 is a bad model
# 1 is a good model
# R squared is 0.647 faily ok fit

# p value should be less than .05  should be included
# p value should be less thans .05 for idv to be significant

# wt is significant
# Intercept is insignificant - can be ignored
predicted_Hwt = 4.0341 * Bwt
# Regression does a hypotisis testing
# H0 ( Null Hypothysis ): No relation between dv and idv
# if p is greater than 0.95 , I accpet null hypothisis with 95 % confidence
# if p is less than 0.95 , I reject null hypothisis with 95 % confidence
# I reject null hyp that there is no relationship

# Accracy of the model ?
training_as_test = catsData.iloc[:,0:2]

# predicting heart for bw present in training data
predicting_hwt_trdata = cats_simple_linear_model.predict(training_as_test)

percent_error = (catsData['Hwt'] - predicting_hwt_trdata)/catsData['Hwt']
abs_percent_error = abs(catsData['Hwt'] - predicting_hwt_trdata)/catsData['Hwt']

model_eval = pd.DataFrame({'Bwt': catsData['Bwt'],
                           'Actual Hwt': catsData['Hwt'],
                            'Predicted Hwt' : predicting_hwt_trdata,
                            'Percentage Error': percent_error,
                            'Abs Percentage Error': abs_percent_error
                           })
# if the error sign is used , positives and negatives will cancel out showing less error
# No recomended and not used
#np.mean(model_eval['Percentage Error']) 
# MEan abs percentage error : (MAPE)
np.mean(model_eval['Abs Percentage Error']) # 89.8% accurate with 11 .2% error
plt.scatter(model_eval['Bwt'],model_eval['Actual Hwt'])
plt.scatter(model_eval['Bwt'],model_eval['Predicted Hwt'],c='red')
# To re verify intercept, red line will touch hwt = -.3567 when bwt = 0
plt.xlim([0,5])
plt.ylim([-1,18])
#############################################################


################ Training - Test Split #####################
## Model eval from test data
# Model Building with trainning data

# Steps in regression analysis : 
# Step 1 : Read file and split data in 70 : 30 ratio for training and test
# Step 2 : Draw a scatter plot with training data
# Step 3 : Using Training data calculate correlation perform correlation analysis
# Step 4 : Buind Regression Model with Training data
           # Find  co eff , Dependent data and intercept value and substitute it in 
           # Ind = m* Dep +C and find R Squared value
# Use Test data from here on
# Step 5 :  Predict independent variable using data from test data using the training model generated in step 4 
# Step 6 : Calcuate MAPE value - (Mean abs percentage error ) abs value of actual - predicted  value
    
# Build the model on traiibg to evaluate on test data
### Stratified Samples : Repesenting all class of data @@@ 
#70 % - Training  30 %  for Test
## 101 for training and 43 to test
## Random sampling data is done
round(0.7*144)
round(0.3*144)
# 101 for training and 43 for testing
aa = np.arange(100)
np.random.seed(12345) # seed is a technique used to generate random numbers # giving number inside will ensure consistent method is used in generating the random numbers
## Fixing the randomness  for reproduablity and could be any int
seventy_percent_data = np.random.choice(aa,70,replace = False) # random samping without repeating values
np.in1d(aa,seventy_percent_data) # present in 70 % data
~np.in1d(aa,seventy_percent_data) # Not in 70 % Data
remaining_thirty_percentage = aa[~np.in1d(aa,seventy_percent_data)]

all_rows = np.arange(catsData.shape[0])
no_tr_sample  = round(0.7*catsData.shape[0])
no_te_sample  = round(0.3*catsData.shape[0])
# 101 for training and 43 for testing
np.random.seed(10)
training_samples = np.random.choice(all_rows,no_tr_sample,replace=False)
testing_samples =  all_rows[~np.in1d(all_rows,training_samples)]

cat_training_data = catsData.iloc[training_samples,:] # 70 % data for training
cat_testing_data = catsData.iloc[testing_samples,:] # 30 % data for testing

plt.scatter("Bwt","Hwt",data=cat_training_data)
plt.xlabel("Body Weight(kg)")
plt.ylabel("Heart Weight(g)")

# Hwt = m* Bwt +C
# Step3: Correlation analysis
# Co varianc helps measuing the releationship between variables
# Co var depends on the scale of data
# corrlelation is scaled co variance ranging -1 and 1
# -1 : String neg corr
# +1: Strong pos corr
# 0 - No corr  Note: -.5 to 0 to +.5 is the scale to inform no corr can be done
## Proceed with reg wien there is reasonably strong corr
np.mean(cat_training_data["Bwt"])
np.var(cat_training_data["Bwt"])
np.std(cat_training_data["Bwt"])
cat_training_data["Bwt"].cov(cat_training_data["Hwt"])
cat_training_data["Bwt"].corr(cat_training_data["Hwt"])
cats_simple_linear_training_model = smf.ols(formula='Hwt ~ Bwt', data = cat_training_data).fit()
cats_simple_linear_training_model.summary()


# R Squareed is the standard metric to measure a good fit
# R squared is 0.696 faily ok fit
## Hwt = m* Bwt +C
# formula : 4.1707 * Bwt -0.7076
#*********# Evaluate with test data
training_as_testData = cat_testing_data.iloc[:,0:2]
# predicting heart for bw present in testing data
predicting_hwt_testdata = cats_simple_linear_training_model.predict(training_as_testData)
abs_percent_error_testData = abs(cat_testing_data['Hwt'] - predicting_hwt_testdata)/cat_testing_data['Hwt']
model_eval_training = pd.DataFrame({'Bwt': cat_testing_data['Bwt'],
                           'Actual Hwt': cat_testing_data['Hwt'],
                            'Predicted Hwt' : predicting_hwt_testdata,
                            'Abs Percentage Error': abs_percent_error_testData
                           })
# if the error sign is used , positives and negatives will cancel out showing less error
# No recomended and not used
#np.mean(model_eval['Percentage Error']) 
# MEan abs percentage error : (MAPE)
np.mean(model_eval_training['Abs Percentage Error']) #14.34
plt.scatter(model_eval_training['Bwt'],model_eval_training['Actual Hwt'])
plt.scatter(model_eval_training['Bwt'],model_eval_training['Predicted Hwt'],c='red')
plt.xlim([0,5])
plt.ylim([-1,18])

####################################################s##########