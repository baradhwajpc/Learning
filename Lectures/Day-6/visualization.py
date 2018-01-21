# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:19:45 2017

@author: baradhwaj
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


os.getcwd()
os.chdir("G:\Python-Learning\Lectures\Day-6")
wg_Data = pd.read_csv("data\\wg.csv")
print(wg_Data)

print(wg_Data['wg'].isnull().sum())
print(wg_Data['metmin'].isnull().sum())
cond1 = wg_Data['wg'].isnull() == True
cond2= wg_Data['metmin'].isnull() == True
print(wg_Data.loc[cond1 | cond2,:].shape[0])

avg_wg = np.mean(wg_Data['wg'])
print(wg_Data.loc[wg_Data['wg']> avg_wg,:].shape[0])

#monthlyGrouped = wg_Data.groupby(['Shift']).mean()
abv_avg = wg_Data.loc[wg_Data['wg']> avg_wg,:]
print(abv_avg.groupby(['Shift']).size())
wgClean = wg_Data.dropna()
print(wgClean)

##############################  Visualization #################

##############################  Histogram#################

histPlt = wgClean['wg'].plot.hist()
histPlt.set_xlabel("Weight Gain")
histPlt.set_ylabel("No of Observations")
histPlt.set_title("Distibution of Weight Gain")
histPlt.set_xlim([5,45])

plt.hist(wgClean['wg'],color="aqua")
plt.xlabel("Weight Gain")
plt.ylabel("No of Observations")
plt.title("Distibution of Weight Gain")
plt.xlim([5,45])

# Comparision of distibution 
wgClean.boxplot(column='wg',by='Gender')
wgClean.boxplot(column='wg',by='Shift')
#custom pinning
wgClean['wg'].plot.hist(bins=[0,20,40,60]) # Pandas
plt.hist(wgClean['wg'],bins=[0,10,20,30,40,50,60,70]) # Matplotlib
plt.hist(wgClean['wg'],bins=[0,20,40,60])
'''


plt.figure()
'''


######################## Box plot ##############################

# Tukeys box plot
a1 = np.random.normal(10,1,100)
plt.hist(a1)
np.mean(a1)
np.median(a1)
 
a2 = np.append(a1,100000)
plt.hist(a2)
np.mean(a2)
np.median(a2)

## IQR 

wgClean['wg'].plot.box()
wgClean['wg'].describe()

Q3 = np.percentile(wgClean['wg'],75)
Q1 = np.percentile(wgClean['wg'],25)
iqr = Q3-Q1
uwl = Q1 + 1.5*iqr
lwl = Q1 - 1.5*iqr




##################### Scatter plot###########################3

scatterPlot = wgClean.plot.scatter('metmin','wg')
scatterPlot.set_xlabel('Activities')
scatterPlot.set_xlabel('Weight Gain')

# matplotlib 
# pltooing 2 arraus / series / list
plt.scatter(wgClean['metmin'],wgClean['wg']) 
plt.scatter('metmin','wg',data=wgClean)

## Add color for male and female observations
wgClean['color'] = 'red'
wgClean.loc[wgClean["Gender"]=="M","color"] = "blue"
plt.scatter('metmin','wg',data=wgClean,c="color")



################ Line plot###########33


os.getcwd()
os.chdir("G:\Python-Learning\Lectures\Day-6")
stock_Data = pd.read_csv("data\\Stock_Price.csv")
print(stock_Data)

stock_plot = stock_Data['DELL'].plot.line()
stock_plot.set_xlabel('Time Instance')
stock_plot.set_ylabel('Closing stock price of dell')

stock_Data.index =  pd.date_range(start='2017-01-01',periods=76,freq='D')
stock_plot = stock_Data['Intel'].plot.line()
stock_plot.set_xlabel('Time Instance')
stock_plot.set_ylabel('Closing stock price of Intel')


stock_plot = stock_Data.plot.line()
stock_plot.set_xlabel('Time Instance')
stock_plot.set_ylabel('Closing stock price of Intel vs Dell')

### If data ttime is provided in index , it becomes time series data
pd.date_range(start='2017-01-01',periods=76,freq='D')
pd.date_range(start='2017-01-01',periods=76,freq='M')

stock_Data.index =  pd.date_range(start='2017-01-01',periods=76,freq='D')
 
stock_plot = stock_Data.plot.line()
stock_plot.set_xlabel('Time Instance')
stock_plot.set_ylabel('Closing stock price of Intel vs Dell')


############## Bar plot #############################3
stock_Data.iloc[0:10,0].plot.bar() # dell
stock_Data.iloc[0:10,1].plot.bar() # Intel
stock_Data.iloc[0:10,:].plot.bar() # Dell vs Intel

stock_Data.iloc[:,:].plot.bar() # Dell vs Intel al daa
