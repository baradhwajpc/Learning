# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:35:24 2017

@author: baradhwaj
"""
import numpy as np

xVec = [42,85,84,23,11,55,14,96,13,30]
yVec = [13,8,85,71, 1,7,55, 2,34,24]
xVec = np.array(xVec)

# Less than 60
condition1 = xVec > 60
print(condition1)
print(xVec[condition1])

# yVec values less than mean of yVec
yVec = np.array(yVec)
condition2 = yVec < np.mean(yVec)
print(condition2)
print(yVec[condition2])

# count of odd numbers in xVec
xVec = [42,85,84,23,11,55,14,96,13,30]
xVec = np.array(xVec)
condition3 = xVec %2!=0
print(condition3)
print(np.count_nonzero(xVec[condition3]))

## find values in yVec between min and max of xVec
xVec = [42,85,84,23,11,55,14,96,13,30]
yVec = [13,8,85,71, 1,7,55, 2,34,24]
xVec = np.array(xVec)
yVec = np.array(yVec)
minVal = np.min(xVec)
maxVal = np.max(xVec)
print(minVal)
print(maxVal)
yVecpos = np.nonzero(np.logical_and(yVec>minVal, yVec<maxVal))
print(yVecpos)
print(yVec[yVecpos])