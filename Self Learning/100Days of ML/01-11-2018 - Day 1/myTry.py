# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:20:31 2018

@author: baradhwaj
"""
import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re 

os.getcwd()
os.chdir('G:\\Python\\Learning\\Self Learning\\100Days of ML\\01-11-2018 - Day 1')
countryData = pd.read_csv("athlete_country.csv",header=0)
athleteData = pd.read_csv("forbes100_2018_athlets.csv",header=0)
countryData.head()
athleteData.head()
countryData.describe()
athleteData.describe()


pay =[]
paySeries = []
endorsment = []
endorsmentSeries = []
salary = []
salarySeries = []


for p in athleteData['Pay']:
    p = p.replace('$','')
    pay.append(p)
athleteData['Pay'] = pd.Series(pay)

for p in athleteData['Endorsements']:
    p = p.replace('$','')
    endorsment.append(p)
athleteData['Endorsements'] = pd.Series(endorsment)


for p in athleteData['Salary/Winnings']:
    p = p.replace('$','')
    salary.append(p)
athleteData['Salary/Winnings'] = pd.Series(salary)



athleteData.PayVal = athleteData.Pay.replace(r'[KM]+$', '', regex=True).astype(float)
type(athleteData.PayVal[0])
aaav = athleteData.PayVal.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1).replace(['K','M'], [10**3, 10**6]).astype(float)


athleteData.PayVal = (athleteData.Pay.replace(r'[KM]+$', '', regex=True).astype(float) *\
              athleteData.Pay.str.extract(r'[\d\.]+([KM]+)', expand=False)
                .fillna(1)
                .replace(['K','M'], [10**3, 10**6]).astype(float))


athleteData['TotalPay'] =  athleteData['Pay']+ athleteData['Endorsements']+ athleteData['Salary/Winnings']
with sns.axes_style("white"):
    plt.subplot(211)
    sns.swarmplot(x="Pay", y="Sport", data=athleteData)
