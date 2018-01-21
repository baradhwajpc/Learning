# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 07:56:50 2017

@author: baradhwaj
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#import datetime
os.getcwd()
os.chdir("G:\Python\Learning\Lectures\Day-5\Homework")
airquality_data = pd.read_csv("data\\airquality.csv")
print(airquality_data)
#1. Get the histogram distribution of Solar.R
# Using Pandas plotting function
hist_plt = airquality_data["Solar.R"].plot.hist()
hist_plt.set_xlabel("Solar Radiation")
hist_plt.set_ylabel("Number of observations")
hist_plt.set_title("Distribution of Solar Radiation")
#hist_plt.set_xlim([5,45])
 # or
# - not working Using matplotlib
plt.hist(airquality_data["Solar.R"],color="green")
plt.xlabel("Solar Radiation")
plt.ylabel("Number of observations")
plt.title("Distribution of Solar Radiation")
#plt.xlim([0,50])
# Q2. Get the boxplot distribution of temperature 
plt.boxplot(airquality_data["Temp"])
# Q3. Generate a scatter plot between temperature and solar.R
# Pandas
sct_plt = airquality_data.plot.scatter("Temp","Solar.R")
sct_plt.set_xlabel("Temp")
sct_plt.set_ylabel("Solar.R")

# Matplotlib
# Plotting 2 independent series/array/list
plt.scatter(airquality_data["Temp"],airquality_data['Solar.R'])
# plotting columns of a data frame
plt.scatter("Temp",'Solar.R',data = airquality_data,c = "red")

#Generate a line plot of Solar.R. Create a date column and use that as x axis in line plot. Note:
#Date and time available in data frame. You may have to refer the documentation of data to know the year.
# May to Sep 1973
## Create index for 
airquality_data.index = pd.date_range(start='1973-05-01', periods=153, freq='D')
solarRData = airquality_data['Solar.R'].plot.line()
solarRData.set_ylabel('Solar.R')

# or 
date_stamp = "1973-" + \
airquality_data["Month"].map(str)+"-"+\
airquality_data["Day"].map(str)

airquality_data.index = pd.to_datetime(date_stamp)

solarRData = airquality_data['Solar.R'].plot.line()
solarRData.set_ylabel('Solar.R')

solarRData = airquality_data['Temp'].plot.line()
solarRData.set_ylabel('Temp')

############################# MT Cars ##############################
os.getcwd()
os.chdir("G:\Python\Learning\Lectures\Day-5\Homework")
mtcars_data = pd.read_csv("data\\mtcars.csv")
print(mtcars_data)

##1. Compare the mpg boxplot distribution of automatic vs manual transmission cars

mtcars_data['mpg'].plot.box(by='vs') # pandas 
pl = mtcars_data.boxplot(column='mpg',by='vs') # matlablib
# 2.Compare the boxplot distribution of mpg of cars by gears and transmission. One mpg 
#distribution box per gear-am combination
mtcars_data.boxplot(column='mpg',by=['am','gear']) # matlablib
#or
fig = plt.figure()
ax1 = plt.subplot(1,2,1)
mtcars_data.boxplot(column='mpg',by='vs',ax=ax1)
ax2 = plt.subplot(1,2,2)
mtcars_data.boxplot(column='mpg',by='gear',ax=ax2)
fig.suptitle('test title', fontsize=20)
plt.show()

# 3. Generate a scatter plot between mpg and weight of the car
scatterPlot = mtcars_data.plot.scatter('mpg','wt')
scatterPlot.set_xlabel('Miles per Gallon')
scatterPlot.set_ylabel('Weight')

# matplotlib 
plt.scatter(mtcars_data['mpg'],mtcars_data['wt']) 
plt.scatter('mpg','wt',data=mtcars_data)