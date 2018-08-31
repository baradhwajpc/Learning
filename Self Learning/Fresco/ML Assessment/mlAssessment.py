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
    knnmodel = knn.fit(X_train, y_train)
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

answers = pd.DataFrame(testdata['outcome'])
answers.to_csv('output.txt', sep=',')

#model_eval_test_data = pd.DataFrame({'Bwt':cat_test_data["Bwt"],                           'Actual Hwt':cat_test_data["Hwt"],                           'Predicted Hwt':predicted_hwt_test_data})
#np.mean(abs(model_eval_test_data['Actual Hwt'] - model_eval_test_data['Predicted Hwt'])/model_eval_test_data['Actual Hwt'])
