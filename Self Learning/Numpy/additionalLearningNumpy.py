# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 07:54:05 2017

@author: baradhwaj
"""
import numpy as np
 # Create a multi dimensional array 
a = np.arange(15).reshape(5,3)
print(a)
a.shape # 3, 5 prints the shape of the array
a.ndim # prints the dimension
a.dtype.name # data type
a.itemsize # Length of one array element in bytes.
a.size #3 total item count


### Array Creation  ########
b = np.array([2,3,4])
c = np.array( [ [1,2], [3,4] ], dtype=complex )
print(c)

### Print 0 , 1  full and empty arrays

zeros = np.zeros((3,4))
print(zeros)

ones = np.ones((5,8))
print(ones)

full = np.full((3,4),7)
print(full)


empty = np.empty((4,4))
print(empty)

#### Floats in numpy - used instead of arrange
floats = np.linspace( 0, 2, 9 )  
print(floats)  

#printing / rendering arrays
oneD = np.arange(6)                         # 1d array
print(oneD)
twoD = np.arange(12).reshape(4,3)           # 2d array
print(twoD)
threeD = np.arange(24).reshape(2,3,4)    
print(threeD)

