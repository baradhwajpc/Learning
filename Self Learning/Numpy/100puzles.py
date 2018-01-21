# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:41:17 2017

@author: baradhwaj
"""
# Q1.
import numpy as np
import sys as s
import calendar

# Q2.
print(np.version.version)
print(np.show_config())
#Q3. Null vector of size 10
zeros = np.zeros((10))
print(zeros)
# Q4.  find the memory size of any array 
Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))
# Q5 .How to get the documentation of the numpy add function from the command line? 
# python -c "import numpy; numpy.info(numpy.add)"
#Q6. Create a null vector of size 10 but the fifth value which is 1
zeros = np.zeros((10))
zeros[4] = 1
print(zeros)
# Q7. Create a vector with values ranging from 10 to 49
print(np.arange(10,50,1))
# Q8 Reverse a vector (first element becomes last)
print(np.arange(10,50,1)[::-1])
# Q9  Create a 3x3 matrix with values ranging from 0 to 8
print(np.arange(9).reshape(3,3))
# Q10. Find indices of non-zero elements from [1,2,0,0,4,0] 
frmArray = [1,2,0,0,4,0]
# print([i for i, frmArray in enumerate(frmArray) if frmArray!=0]) - fast
print(np.nonzero(frmArray))
# Q 11 Create a 3x3 identity matrix 
print(np.eye(3))
# Q 12 . Create a 3x3x3 array with random values
np.random.random((3,3,3))
# Q 13 Create a 10x10 array with random values and find the minimum and maximum values
tenArr = np.random.random((10,10))
print(tenArr.min())
print(tenArr.max())
#14. Create a random vector of size 30 and find the mean value 
print(np.mean(np.random.randint(30)))

# Q 15 .Create a 2d array with 1 on the border and 0 inside
borderZero = np.ones((10,10))
borderZero[1:-1,1:-1] = 0 # doubtful
print(borderZero)

# Q 16. How to add a border (filled with 0's) around an existing array?
np.pad(borderZero, (0, 1024 - len(borderZero)%1024), 'constant')
# Q 17. What is the result of the following expression? (★☆☆)
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
#Q 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
print(np.diag(np.arange(5), k=0))
# Q 19 19. Create a 8x8 matrix and fill it with a checkerboard pattern
zeroM = np.zeros((8,8),dtype=int)
zeroM[1::2,::2] = 1  # starsat:endsbefore:skip
zeroM[::2,1::2] = 1
print(zeroM)

# 20 Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? np.unravel_index
#print()
arr336 = np.arange(336).reshape(6,7,8)
np.unravel_index(100,arr336.shape)

# 21.Create a checkerboard 8x8 matrix using the tile function 
tileA = np.array([0,1])
print(np.tile(tileA,(8,8)))

# 22.  Normalize a 5x5 random matrix
rand5matrix= np.random.random((5,5))
xmax, xmin = rand5matrix.max(), rand5matrix.min()
rand5matrix = (rand5matrix - xmin)/(xmax - xmin)
print("After normalization:")
print(rand5matrix)

# 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA)
color =  np.dtype(['r',np.ubyte,1],
                  ['g',np.ubyte,1],
                  ['b',np.ubyte,1],
                  ['a',np.ubyte,1])


# Q24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) 
x= np.random.random((5,3))
y= np.random.random((3,2))
print(np.dot(x,y))
#or
x = np.arange(15).reshape((5, 3))
y = np.arange(6).reshape((3, 2))
np.dot(x, y)

#Q25. Given a 1D array, negate all elements which are between 3 and 8, in place.               
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)

#Q26.  What is the output of the following script? (★☆☆)
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

# Q27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)¶
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z # invalid

#28. What are the result of the following expressions?
np.array(0) / np.array(0) # nan
np.array(0) // np.array(0)# 0 
np.array([np.nan]).astype(int).astype(float) # array

# Q29.How to round away from zero a float array
rand = np.random.uniform(-10,10,10)
print(np.copysign(np.ceil(np.abs(rand)),rand))

# Q 30. Find common elements in 2 arrays
aa= np.random.randint(0,10,10)
bb= np.random.randint(0,10,10)
print(np.intersect1d(aa,bb))

# Q.31 Numpy raise error
old_settings = np.seterr(all='warn', over='raise')
#Q32. Is the following expression true
np.sqrt(-1) == np.emath.sqrt(-1)
# Q33. How to get the dates of yesterday, today and tomorrow?
from datetime import date,timedelta
print(date.today() - timedelta(1))
print(date.today() + timedelta(1))
#Q34. How to get all the dates corresponding to the month of July 2016? 
print(np.arange('2016-07', '2016-08', dtype='datetime64[D]'))
#Q35.How to compute ((A+B)*(-A/2)) in place (without copy)
# np.add(out=), np.negative(out=), np.multiply(out=), np.divide(out=)
a=5
b=6
print(np.multiply(np.add(a,b),np.divide(np.negative(a),2)))
# Q36. Extract the integer part of a random array using 5 different methods
Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))

#Q37. Create a 5x5 matrix with row values ranging from 0 to 4 
a = np.zeros((5,5))
a+=np.arange(5)
print(a)

#38. Consider a generator function that generates 10 integers and use it to build an array
iterable = (x*x for x in range(10))
print(iterable)
np.fromiter(iterable,dtype = float,count=-1)

#Q39. Create a vector of size 10 with values ranging from 0 to 1, both excluded 
np.linspace(0,1,endpoint=False)
#Q40. Create a random vector of size 10 and sort it 
xRand10 = np.random.random(10)
xRand10.sort()
print(xRand10)

#Q41. How to sum a small array faster than np.sum?
tenelms = np.arange(10)
np.add.reduce(tenelms)

#Q42.Consider two random array A and B, check if they are equal (★★☆)
aArr = np.random.randint(0,2,5)
bArr = np.random.randint(0,2,5)
np.array_equal(aArr,bArr)

#43. Make an array immutable 
immutArr = np.zeros(10)
immutArr.flags.writeable = False
immutArr[1] = 4
        
#Q44. Consider a random 10x2 matrix representing cartesian coordinates, 
#convert them to polar coordinates (★★☆)¶
#(hint: np.sqrt, np.arctan2)
randArr = np.random.random((10,2))
x,y = randArr[:,0] , randArr[:,1] 
t = np.arctan2(y,x)
print(t)

#45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
#(hint: argmax)
rand10 = np.random.randint(10)
np.argmax(rand10,axis=0)
