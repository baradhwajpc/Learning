# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:26:38 2017

@author: baradhwaj
"""

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# Q1. Read the csv file and check the data types. Note that certain columns has numbers with
#commas in between which might have been read as a non-numeric data type. You can't
#just convert the data type; it will then have junk numbers. You have to remove commas.
# File Read
os.getcwd()
os.chdir("G:\Python\Learning\Lectures\Day-8\Homework")
empData = pd.read_csv("data\\ssamatab1.csv",thousands =',')
print(empData)

#Q2. Which Area had the highest unemployment rate in December 2015?
decValues = empData.loc[(empData['Month']==12) &(empData['Year']==2015),:]
decValues.loc[decValues['Unemployment Rate'] == np.max(decValues['Unemployment Rate']),'Area']

#Q3. Which area had the highest ever unemployment rate and when did that happen? 
mostUnemployed = np.max(empData['Unemployment Rate'])
empData.loc[empData['Unemployment Rate'] ==mostUnemployed,["Area","Year","Month"]]

#Q4. Which state had the highest ever unemployment rate and when did that happen?
mostUnemployed = np.max(empData['Unemployment Rate'])
empData.loc[empData['Unemployment Rate'] ==mostUnemployed,["State FIPS Code","Year","Month"]]
# Calculate avg unemploymet of each state


#Q5.Obtain Yearly Unemployment rate by aggregating the data. One way would be to take
#average of unemployment rate column directly. But that's not mathematically right. You
#need to sum up the Unemployed and Civilian labor force by Year and then calculate the
#ratio for calculation of Unemployment rate
empData_GroupYear = empData.groupby("Year")
yearlySums = empData_GroupYear["Civilian Labor Force","Unemployment"].agg(np.sum)
yearlyupRate = yearlySums['Civilian Labor Force']/yearlySums['Unemployment']


# Q6 . For State

empData_GroupState = empData.groupby("State FIPS Code")
stateSums = empData_GroupState["Civilian Labor Force","Unemployment"].agg(np.sum)
stateupRate = stateSums['Civilian Labor Force']/stateSums['Unemployment']

#Q7. Plot the histogram and boxplot of unemployment rate

# Using matplotlib
plt.hist(empData["Unemployment Rate"],color="red",bins = [0,3,6,9,12,15,18,21,24,27,30])
plt.xlabel("Unemployment Rate")
plt.ylabel("Number of observations")
plt.title("Distribution of Unemployment Rate")
# box plot
plt.boxplot(empData["Unemployment Rate"])

#Q8.Compare the boxplot distribution of unemployment rate between top 4 states with highest
#civilian labor force
empDataBox = empData_GroupState["Civilian Labor Force","State FIPS Code"].agg(np.max)
empDataSorted = empDataBox.sort_values("Civilian Labor Force",ascending=False).head(4)
top4StateAllData = empData.loc[empData['State FIPS Code'].isin(empDataSorted['State FIPS Code']),:]
#daysGreat = airquality_data.loc[cond ,["Day","Solar.R"]]
top4StateAllData.boxplot(column="Unemployment Rate",by="State FIPS Code")

# Q9. Visualize the relationship between civilian labor force and unemployment rate using scatter plot
ur_grp = empData.groupby(['Unemployment Rate'])
cv_data = ur_grp[["Civilian Labor Force"]].agg(np.sum)
cv_data['URate'] = cv_data.index
plt.scatter(cv_data['Civilian Labor Force'],cv_data['URate'])
#Q10.  Draw line plot of yearly unemployment rate of US (Year in xaxis and unemployment rate
#of US in yaxis)
lineUnemployment =yearlyupRate.plot.line()
