# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:33:27 2017

@author: baradhwaj
"""
'''
Create a list of birth years of 5 friends/family member (e.g: br_yr = [1986, 1989, 1975, 1981,
1978]). Calculate their age (years alone) as of 2017 using 3 approaches and save it a list. Assume
their birth days are over in 2017
 Regular for loops
 List comprehension
 Vectorized operation using numpy array
'''
import numpy as np
import pandas as pd
# Regular
yob = [1986, 1989, 1975, 1981,1978]
age = [0.0]*len(yob)
for i in enumerate(yob):
    val = i[1]
    pos = i[0]
    age[pos] = 2017 - val
print(age)

# Lis Comprehension
ageList = [2017 - i for i in yob]
print(ageList)
## VECTORIZED OPERATION/NUMPY OPERATION
ageVectorized = 2017 -np.array(yob)
print(ageVectorized)

'''
2.Create a string “this is a python exercise which is neither too easy nor too hard to be solved in
the given amount of time”. Split the string to list of individual words [Hint: split command. Don’t
search in classwork]. Remove words like ‘is’, ‘a’ and ‘the’ programmatically using 3 approaches
Regular for loops
List comprehension
Vectorized operation using numpy array
'''
ipstr = 'this is a python exercise which is neither too easy nor too hard to be solved in the given amount of time'
strArr = ipstr.split()
print(strArr)
newStr = []
for item in strArr:
    print(item)
    if not (item == 'a' or item == "is" or item == "the"):
        print(item,'true')
        newStr.append(item)
print(newStr)
print(len(newStr))
# List Comprehension
truncated = [i for i in strArr if i != 'a' and i !='is' and i != 'the']
print(truncated)

# vectorized
ipA = np.array(strArr)
newStr = ipA[(ipA!='a')&(ipA!='is')&(ipA!='the')]
print(newStr)