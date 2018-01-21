# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 08:21:05 2017

@author: baradhwaj
"""
## Diwali assignment
import numpy as np 
import pandas as pd
import os
#import datetime
# Q1. Read data from file 
#Q2 . Understand the data . 
os.getcwd()
os.chdir("G:\Python-Learning\Lectures\Day-5\Homework")
airquality_data = pd.read_csv("data\\airquality.csv")
print(airquality_data)

# Q3,Q4 . No of rows and cols
print(airquality_data.shape[0])
print(airquality_data.shape[1])

# Q5 print all column names
print(airquality_data.columns)

# Q 6 . Print count of nulls in ozone column
print(airquality_data['Ozone'].isnull().sum())

# Q7 .Slice from airquality a dataframe which only has rows with valid entries for Solar.R. Remove rows which has null values in Solar.R column
solarNotNull = airquality_data[np.isfinite(airquality_data['Solar.R'])]
print(solarNotNull)

# Q 8 . Avg value of ozone 
print('Avg value of Ozone : ', (airquality_data['Ozone'].sum()) / airquality_data['Ozone'].count())
# OR 
print('Avg value of Ozone : ', np.mean(airquality_data['Ozone'].mean()))
# Q9. What is the average value of Solar.R on days with temperature above average temperature? 
avg_temp  = np.mean(airquality_data['Temp'])
cond = airquality_data['Temp'] >avg_temp
print(cond)
daysGreat = airquality_data.loc[cond ,["Day","Solar.R"]]
print(daysGreat)
print('average value of Solar.R on days with temperature above average temperature: ' , np.mean(daysGreat['Solar.R']))

s2 = airquality_data['Temp'].mean()
condn1 = airquality_data['Temp'] > s2
final_avg = airquality_data.loc[condn1,'Solar.R'].mean()


#Q10. Slice only records of 15th day of each month 
fifteenRec = airquality_data['Day'] == 15
print(airquality_data.loc[fifteenRec ,:])

# Q 11. Slice records of 6th and 8th month alone 
sixth = airquality_data['Month'] == 6
eighth = airquality_data['Month'] == 8
print(airquality_data.loc[(airquality_data['Month'] == 6) | (airquality_data['Month'] == 8),:])

# Q12. What is the average ozone values of the days where both Solar.R and Temperature are above their averages? 
solarMean = np.mean(airquality_data['Solar.R'].mean())
tempMean = np.mean(airquality_data['Temp'].mean())
solarCond = airquality_data['Solar.R'] >solarMean
tempCond  = airquality_data['Temp'] >tempMean

ozoneGreater = airquality_data.loc[solarCond & tempCond,["Day","Ozone"]]
print ('Average ozone values of the days where both Solar.R and Temperature are above their averages: ' , 
       np.mean(ozoneGreater['Ozone']))

## For loops
# Q 13 .   Calculate average values of Ozone, Solar, 
  #Wind and Temperature and save in a list/array/series  
  ### wrong approach --  take mean instead of fillna with 0 
  
# optimized data processing:
print(airquality_data[["Ozone","Solar.R","Wind","Temp"]].apply(np.mean))

#loops:
length = len(airquality_data.index)
ozoneSum = 0
solarSum = 0
windSum = 0
tempSum = 0
print(airquality_data)


for item, row in airquality_data.iteritems():
            ozoneSum+=row['Ozone']
            solarSum+=row['Solar.R']
            windSum +=row['Wind']
            tempSum +=row['Temp']
# Excel does not take into account the values with nan
allAvg = pd.Series()
allAvg.set_value('Ozone',np.mean(airquality_data['Ozone']))
allAvg.set_value('Solar.R',np.mean(airquality_data['Solar.R']))
allAvg.set_value('Wind',np.mean(airquality_data['Wind']))
allAvg.set_value('Temp',np.mean(airquality_data['Temp']))
print(allAvg)
#print('Ozone Avg: ',ozoneSum / length)
#print('Solar Avg: ',solarSum / length)
#print('Wind Avg: ',windSum / length)
#print('Temp Avg: ',tempSum / length)

# Q 14  Calculate month-wise average Ozone and save in a list/array/series 

# optimized data processing:
airquality_data.groupby(['Month']).mean()
# without for loop:
monthlyGrouped = airquality_data.groupby(['Month']).mean()
ozoneMonthlyData = pd.Series(monthlyGrouped['Ozone'])
print(ozoneMonthlyData)

# with for loop
ozoneSeries = pd.Series()
for x in airquality_data['Month'].unique():
    ozoneData = (airquality_data['Ozone'][airquality_data['Month'] == x]).mean()
    #print(ozoneData)
    ozoneSeries.set_value(x,ozoneData)
print(ozoneSeries)
     
# Q15. Calculate month-wise average Ozone, Solar, Wind and Temperature and save in a matrix/data 

# optimized Data processing:
airquality_dataM = airquality_data.groupby("Month")
print(airquality_dataM[["Ozone",'Solar.R','Wind','Temp']].agg(np.mean))
    
ozoneAvg = pd.Series()
solarAvg = pd.Series()
windAvg = pd.Series()
tempAvg = pd.Series()

for x in airquality_data['Month'].unique():
    ozoneData = (airquality_data['Ozone'][airquality_data['Month'] == x]).mean()
    solarData= (airquality_data['Solar.R'][airquality_data['Month'] == x]).mean()
    windData = (airquality_data['Wind'][airquality_data['Month'] == x]).mean()
    tempData = (airquality_data['Temp'][airquality_data['Month'] == x]).mean()
    
    ozoneAvg.set_value(x,ozoneData)
    solarAvg.set_value(x,solarData)
    windAvg.set_value(x,windData)
    tempAvg.set_value(x,tempData)
    
    
    data = {'ozoneAvg': ozoneAvg,
            'solarAvg': solarAvg,
            'windAvg': windAvg,
            'tempAvg': tempAvg
            }
avgsData = pd.DataFrame(data)
print(avgsData)
    
######################## end of air polluttion ###############################
# file reading
mtcars_data = pd.read_csv("data\\mtcars.csv")
print(mtcars_data)

# Data Frame properties and quality check 
# Q3,Q4 . No of rows and cols
print(mtcars_data.shape[0])
print(mtcars_data.shape[1])
# Q5 print all column names
print(mtcars_data.columns)
# use describe
mtcars_data.describe()

# Data Frame slicing 
# Q7.Average miles per gallon (mpg) of all cars 
 print('All car mpg mean: ',np.mean(mtcars_data['mpg']))
 
# q8. Average mpg of automatic transmission cars  - 0 
condAuto = mtcars_data['am'] ==0
print('Avg mpg of auto tran cars: ',np.mean(mtcars_data['mpg'][condAuto]))

#Q9. Average mpg of manual transmission cars 
condManual = mtcars_data['am'] ==1
print('Avg mpg of manual tran cars: ',np.mean(mtcars_data['mpg'][condManual]))

# Q10 . Average Displacement of cars with 4 gears 
condGear =  mtcars_data['gear'] ==4
print('Average Displacement of cars with 4 gears: ' ,np.mean(mtcars_data['disp'][condGear]))

# Q 11.  Average Horse power of cars with 3 carb 
congCarbs3 =  mtcars_data['carb'] ==3
print('# Q 11.  Average Horse power of cars with 3 carb: ',np.mean(mtcars_data['hp'][congCarbs3]))

 #Q12. Average mpg of automatic cars with 4 gears
#condAuto & condGear
print('Average mpg of automatic cars with 4 gears: ', np.mean(mtcars_data['mpg'][condAuto & condGear]))

# Q13.  Average qsec of cars with mpg above average mpg and weight below average weight 
avgMPG = np.mean(mtcars_data['mpg'])
condGreaterMPG = mtcars_data['mpg'] > avgMPG

avgWT = np.mean(mtcars_data['wt'])
congLessWGT = mtcars_data['wt'] > avgWT

print(mtcars_data.loc[condGreaterMPG ,['qsec']])
print(mtcars_data.loc[congLessWGT ,['qsec']])

print('Average qsec of cars with mpg above average mpg and weight below average weight: ' 
      ,np.mean(mtcars_data.loc[condGreaterMPG & congLessWGT ,['qsec']]))

print('Average qsec of cars with mpg above average mpg and weight below average weight: ' 
      ,np.mean(mtcars_data.loc[condGreaterMPG | congLessWGT ,['qsec']]))

#Q14 Entire row of the vehicle which has the highest miles per gallon 
maxmpg = np.max(mtcars_data['mpg'])
print(pd.Series(mtcars_data['gear'][mtcars_data['mpg']==maxmpg]))

# Q15.  Entire row of vehicle with the highest horsepower 
maxHP = np.max(mtcars_data['hp'])
print('Entire row of vehicle with the highest horsepower: ', mtcars_data[mtcars_data['hp']==maxHP])

# Q 16 . Mileage and hp of car with highest weight  - Shld ask clarification
maxWGT = np.max(mtcars_data['wt'])
milnHP = pd.DataFrame(mtcars_data[['mpg', 'hp']][mtcars_data['wt']==maxWGT]).values[0]
print('Mileage and hp of car with highest weight : ',milnHP)

# Q 17. Calculate ratio of mpg to carb for each car and calculate the average of ratio 
 print(np.mean(mtcars_data['mpg']/mtcars_data['carb']))

 #Q 18.  Weight of the car with the minimum displacement 
mindisp = np.min(mtcars_data['disp'])
print('Weight of the car with the minimum displacement : ',pd.Series(mtcars_data[['wt']][mtcars_data['disp']==mindisp].values[0]))

# Q 19. . Slice all columns of 3 gear cars 
gearCond  = mtcars_data['gear'] == 3
print(mtcars_data.loc[gearCond,:])

# Q 20. Slice mpg, displacement and hp columns of manual transmission cars 

print('Slice mpg, displacement and hp columns of manual transmission cars ' , mtcars_data.loc[condManual ,['mpg','disp','hp']])


#Q21. What is average mpg of 3, 4 and 5 gear cars. Save output as a list/array/series 
mpgSeries = pd.Series()
for x in mtcars_data['gear'].unique():
    mpgData = (mtcars_data['mpg'][mtcars_data['gear'] == x]).mean()
    mpgSeries.set_value(x,mpgData)
mpgSeries.sort_index(inplace=True)
print('Average mpg of 3, 4 and 5 gear cars. Save output as a list/array/series: ' , mpgSeries)

#Q22. What is average hp, average wt, average qsec, average vs for 3, 4 and 5 gear cars. 
#Save output as a matrix or data frame 
  
df = pd.DataFrame()

hpAvg = pd.Series()
wtAvg = pd.Series()
qsecAvg = pd.Series()
vsAvg = pd.Series()

for x in mtcars_data['gear'].unique():
    hpData = (mtcars_data['hp'][mtcars_data['gear'] == x]).mean()
    wtData= (mtcars_data['wt'][mtcars_data['gear'] == x]).mean()
    qsecData = (mtcars_data['qsec'][mtcars_data['gear'] == x]).mean()
    vsData = (mtcars_data['vs'][mtcars_data['gear'] == x]).mean()
    
    hpAvg.set_value(x,hpData)
    wtAvg.set_value(x,wtData)
    qsecAvg.set_value(x,qsecData)
    vsAvg.set_value(x,vsData)
    
    df = {'hpAvg': hpAvg,
            'wtAvg': wtAvg,
            'qsecAvg': qsecAvg,
            'vsAvg': vsAvg
            }
avgsCarData = pd.DataFrame(df)
avgsCarData.sort_index(inplace=True)
print(avgsCarData)


############################## End of second asssement ###############################################