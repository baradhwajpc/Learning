# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 07:46:30 2018

@author: baradhwaj
"""

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
import sklearn.preprocessing as preprocessing
import seaborn as sns
# Step 0 :  Read data from csv file 
os.getcwd()
os.chdir("G:\Python\Learning\Self Learning\Fresco\ML Assessment")
rawdata = pd.read_csv("train.csv")
earnings = rawdata.dropna(subset=['occupation'])
# Step-1 : Data Cleanup

workclass_lbls = ['State-gov','Private','Self-emp-inc','Self-emp-not-inc','Federal-gov',
          'State-gov','Without-pay','Never-worked','Local-gov']
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(workclass_lbls)
wc_labelencoded = labelencoder.transform(earnings.workclass)
earnings['workclass_num'] = wc_labelencoded
## removing this column 

maritalstatus_lbl = ['Divorced','Never-married','Married-civ-spouse','Widowed',
                      'Separated','Married-spouse-absent','Married-AF-spouse'] # 0-6
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(maritalstatus_lbl)
ms_labelencoded = labelencoder.transform(earnings['marital-status'])
earnings['marital_num'] = ms_labelencoded

relationship_lbl = ['Not-in-family','Husband','Wife','Own-child','Other-relative','Unmarried'] # 0 - 5
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(relationship_lbl)
r_labelencoded = labelencoder.transform(earnings['relationship'])
earnings['race_num']  = r_labelencoded

race_lbl = ['White','Black','Amer-Indian-Eskimo','Asian-Pac-Islander','Other'] # 0 - 4
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(race_lbl)
race_labelencoded = labelencoder.transform(earnings['race'])
earnings['rel_num'] =  race_labelencoded

gender_lbl = ['Male','Female'] # 0-1
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(gender_lbl)
gender_labelencoded = labelencoder.transform(earnings['gender'])
earnings['sex_num'] = gender_labelencoded


X = earnings[['workclass_num', 'educational-num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital-gain', 'capital-loss']]
y = earnings['income_>50K']

hmap = earnings.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True);

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import KFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
from sklearn.linear_model import LogisticRegression

# instantiate
logreg = LogisticRegression()
# fit
logModel = logreg.fit(X_train, y_train)
# predict
y_pred = logreg.predict(X_test)
print('LogReg %s' % metrics.accuracy_score(y_test, y_pred))

kf = KFold(len(earnings), n_folds=10, shuffle=False)
print('KFold CrossValScore %s' % cross_val_score(logreg, X, y, cv=kf).mean())

from sklearn.neighbors import KNeighborsClassifier
k_range = np.arange(1, 20)
#10,15 -  8 , 0.8539, 
#25 - 0.8572 - 20 , 
# 20 - 18 0.8566 
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores.index(max(scores)), max(scores))

testdata = pd.read_csv("test.csv")



workclass_lbls = ['State-gov','Private','Self-emp-inc','Self-emp-not-inc','Federal-gov',
          'State-gov','Without-pay','Never-worked','Local-gov']
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(workclass_lbls)
wc_labelencoded = labelencoder.transform(testdata.workclass)
testdata['workclass_num'] = wc_labelencoded
## removing this column 

maritalstatus_lbl = ['Divorced','Never-married','Married-civ-spouse','Widowed',
                      'Separated','Married-spouse-absent','Married-AF-spouse'] # 0-6
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(maritalstatus_lbl)
ms_labelencoded = labelencoder.transform(testdata['marital-status'])
testdata['marital_num'] = ms_labelencoded

relationship_lbl = ['Not-in-family','Husband','Wife','Own-child','Other-relative','Unmarried'] # 0 - 5
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(relationship_lbl)
r_labelencoded = labelencoder.transform(testdata['relationship'])
testdata['race_num']  = r_labelencoded

race_lbl = ['White','Black','Amer-Indian-Eskimo','Asian-Pac-Islander','Other'] # 0 - 4
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(race_lbl)
race_labelencoded = labelencoder.transform(testdata['race'])
testdata['rel_num'] =  race_labelencoded

gender_lbl = ['Male','Female'] # 0-1
labelencoder = preprocessing.LabelEncoder()
labelencoder = labelencoder.fit(gender_lbl)
gender_labelencoded = labelencoder.transform(testdata['gender'])
testdata['sex_num'] = gender_labelencoded


testX = testdata[['workclass_num', 'educational-num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital-gain', 'capital-loss']]
outcome = logModel.predict(testX)

testdata['outcome'] = outcome



#model_eval_test_data = pd.DataFrame({'Bwt':cat_test_data["Bwt"],                           'Actual Hwt':cat_test_data["Hwt"],                           'Predicted Hwt':predicted_hwt_test_data})
#np.mean(abs(model_eval_test_data['Actual Hwt'] - model_eval_test_data['Predicted Hwt'])/model_eval_test_data['Actual Hwt'])

























# Step 2 : Identify dependent and idv 
# DV -> income_>50k - people with income > 50k
# IDV -> x1,x2,x3,x4s Composition of key ingredents  
# Step -2 : Visualize the data
plt.scatter('x1','y',data=earnings) # x increases y increasess ( noisy data)
plt.scatter('x2','y',data=earnings) # x incr y incre ( better releated)
plt.scatter('x3','y',data=earnings) # x3 increase y decrease ( noisy)
plt.scatter('x4','y',data=earnings) #x4 increase y decrease (better related)

axes = pd.tools.plotting.scatter_matrix(earnings,alpha=0.8)

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
