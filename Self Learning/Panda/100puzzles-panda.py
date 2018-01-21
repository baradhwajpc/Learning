# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 07:48:13 2017

@author: baradhwaj
"""
#Objectives : (indexing, grouping, aggregating, cleaning)

#1.Category : Getting started and  checking panda setup
#Q1: Import panda 
import pandas as pd
import numpy as np
#Q2 :Print the version of pandas that has been imported.
pd.__version__
# Q3 : Print out all the version information of the libraries that are required by the pandas library.
print(pd.show_versions())

# 2. DataFrame basics
# A few of the fundamental routines for selecting, sorting, adding and aggregating data in DataFrames
#Input:
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# Q4 : reate a DataFrame df from this dictionary data which has the index labels.
df = pd.DataFrame(data,index=labels)
# Q5 :  Display a summary of the basic information about this DataFrame and its data.
df.info()
# Q6 : Return the first 3 rows of the DataFrame df.
df[0:3]
# Q 7 Select just the 'animal' and 'age' columns from the DataFrame df.
print(df[['animal','age']])
#Q8. Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].
print(df.loc[df.index[[3, 4, 8]], ['animal', 'age']])
# Q9 Select only the rows where the number of visits is greater than 3.
greaterThan3 = df["visits"]>3
df3 = df[greaterThan3]
print(df3)
# Q 10 Select the rows where the age is missing, i.e. is NaN.
isNan = df["age"].isnull()
dfNan = df[isNan]
print(dfNan)
# Q 11 Select the rows where the animal is a cat and the age is less than 3.
df[(df['animal'] == 'cat') & (df['age'] < 3)]
# Q12 Select the rows the age is between 2 and 4 (inclusive).
df[(df['age'] >2) &(df['age'] <=4)]
# Q13 . Change the age in row 'f' to 1.5.
df.loc['f',:'age'] = 1.5
# Q 14 Calculate the sum of all visits (the total number of visits).
df['visits'].sum()
# Q 15 Calculate the mean age for each different animal in df.
df.groupby("animal")["age"].mean()
#Q16 Append a new row 'k' to df with your choice of values for each column. Then delete that row to return the original DataFrame.
df.loc[-1] = ('k')
df = df.drop(df.index[-1])
print(df)
## Count the animal type
df["animal"].value_counts()
# Q 18 Sort df first by the values in the 'age' in decending order, then by the value in the 'visit' column in ascending order.
 df.sort(['age', 'visits'] ,  ascending=([0, 1])) # or 
 df.sort_values(by=['age', 'visits'], ascending=[False, True])
# Q 19  The 'priority' column contains the values 'yes' and 'no'. 
#Replace this column with a column of boolean values: 'yes' should be True and 'no' should be False.
d = {'yes': True, 'no': False}
df['priority'].map(d)
# Q 20 In the 'animal' column, change the 'snake' entries to 'python'.
df['snake',:'animal'] = 'python'
df['animal'] = df['animal'].replace('snake','python')
print(df)
# Q 21. . For each animal type and each number of visits, find the mean age.
#         In other words, each row is an animal, each column is a number of visits and the values are the mean ages (hint: use a pivot table).
print(df.pivot_table(df, index=['animal','visits'], aggfunc='mean'))

#DataFrames: beyond the basics
#Slightly trickier: you may need to combine two or more methods to get the right answer
#Q 22. You have a DataFrame df with a column 'A' of integers. For example:
 dfInt = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
 print(dfInt.loc[dfInt['A'].shift()!=dfInt['A']])
# Q 23. 23. Given a DataFrame of numeric values, say
#how do you subtract the row mean from each element in the row?
df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
print(df.sub(df.mean(axis = 1),axis= 0))
# 24. Suppose you have DataFrame with 10 columns of real numbers, for example:
   #Which column of numbers has the smallest sum? (Find that column's label.)
dfRand = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
print(dfRand.sum().idxmin())

 # Q 25. How do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?
 print(len(df.drop_duplicates(keep=False)))
 
 #Q26.26. You have a DataFrame that consists of 10 float columns. 
 #Exactly 5 entries in each row are NaN values. 
 #For each row of the DataFrame, find the column which contains the third NaN value.

 dfRnd = pd.DataFrame(np.random.random(size=(10,10)))
 print((dfRnd.isnull().cumsum(axis=1) == 3).idxmax(axis=1))