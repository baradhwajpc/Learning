# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:21:16 2018

@author: baradhwaj
"""

# Pilot
import pandas as pd
import numpy as np
age = pd.Series(np.random.randint(18,70,100))
custID = pd.Series(list(range(100)))
aa = np.arange(1000000)
np.random.seed(12345)
salary = pd.Series(np.random.randint(100,3000,100)*500)
paymentRemarks = pd.Series(np.random.randint(50,56,100))
teliorDepts = pd.Series(np.random.randint(0,10000,100))
homeAvailable = pd.Series(np.random.randint(0,2,100))
creditBalence = pd.Series(np.random.randint(0,10000,100)*50)
empStatus = pd.Series(np.random.randint(0,4,100))
listSeries = [custID,age,salary,paymentRemarks,teliorDepts,homeAvailable,creditBalence,empStatus]

dfCust = pd.concat([custID,age,salary,paymentRemarks,teliorDepts,homeAvailable,creditBalence,empStatus],axis =1)
dfCust.rename(columns={0: 'CustID', 1: 'Age',2:'Salary',3:'PaymentRemarks',4:'TeliorDepts',
                   5:'homeAvailable',6:'CreditBalence',7:'EmploymentStatus'}, inplace=True)
dfCust['Immigration'] = dfCust.Salary > 9000000

cond1 = dfCust["CreditBalence"] >= dfCust["Salary"]
cond2 = (dfCust["EmploymentStatus"] == 2 )
cond3 = (dfCust["EmploymentStatus"] == 3)
dfCust['Bankrupt']  =cond1 & cond2 | cond3
import os
os.getcwd()
dfCust.to_csv('generatedData.csv')