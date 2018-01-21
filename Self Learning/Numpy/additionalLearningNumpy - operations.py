# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 07:54:05 2017

@author: baradhwaj
"""
import numpy as np
## Basic operations 

a = np.array( [20,30,40,50] )
b = np.arange( 4 )
c = a-b
print(c) # subraction
print(b**2) ## square
print(10*np.sin(a)) # complex ops
print(a<35) # condition
print(a*b) #  element wise product
print(a.dot(b)) # matrix product
# Prints random decimals - Upcasting - Adding float and int will result in float
random = np.random.random((2,3))
print(random)
someMatrix = np.ones((2,3), dtype=int)
print(someMatrix)
someMatrix*=3
print(someMatrix)
someRandomMatrix = random + someMatrix
print(someRandomMatrix)

#unary ops  sum , min and max
print(random.sum())
print(random.max())
print(random.min())


#multidim## Axis operator in multidimensional array 
multiDimSomeMatrix  = np.arange(12).reshape(3,4)
print(multiDimSomeMatrix.sum(axis=0))
print(multiDimSomeMatrix.cumsum(axis=1))

 # Universal operations - sqrt, exp , sin ,cos
 print(np.sqrt(multiDimSomeMatrix))
 print(np.exp(multiDimSomeMatrix))
 print(np.sin(multiDimSomeMatrix))
 print(np.cos(multiDimSomeMatrix))
 
 ### Slicing , inexing  -  array
 # Reversing an one dimensional numpy
 print(someRandomMatrix[ : :-1])
 
 
 ### Multidimesional  and functions - use of axis
 def f(x,y):
     return 10*x+y

customFunc = np.fromfunction(f,(5,4),dtype=int) 
print(customFunc)
## print all items in 1 column
print(customFunc[0:5,1]) # or 
print(customFunc[:,1]) 
#When fewer indices are provided than the number of axes, the missing indices are considered complete slices:
print(customFunc[1])  # no axis provided

### Changing the shape of he array
shapeArr = np.floor(10*np.random.random((3,4)))
print (shapeArr.shape)
print(shapeArr.ravel())## prints a flat array
print(shapeArr.reshape(6,2)) # reshapes
print(shapeArr.T) # transpose
print (shapeArr.shape)
## resshape  function returns its argument with a modified shape, whereas the ndarray.resize method modifies the array itself:

# Stacking of arrays
#print(np.vstack((someRandomMatrix,shapeArr))) ## Cannot stack coz, all inputs must match exactly
print(np.vstack((shapeArr,multiDimSomeMatrix))) # 1 d array
print(np.column_stack((multiDimSomeMatrix,shapeArr)))  # n d array

# complex  ful for creating arrays by stacking numbers along one axis. They allow the use of range literals (”:”)
np.r_[1:4,0,4]
## split arrays : hsplit