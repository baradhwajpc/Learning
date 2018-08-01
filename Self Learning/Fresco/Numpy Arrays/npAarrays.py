# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 06:53:11 2018

@author: baradhwaj
"""

import numpy as np
z = np.eye(2)
print(z)
print(np.linspace(1, 10, 5))
print(np.arange(1, 10, 5))
np.random.rand(3, 2)
np.random.random(3, 2)

x1 = np.array([[[-1,1],[-2,2]],[[-3, 3], [-4, 4]]])
x1.shape
x1.ndim
x1.size
x2 = np.ones((3,2,2))
x3 = np.identity(4)
x3 = np.identity(3,4,2)
x4 = np.random.uniform(low = 0,high=1,size = (3,4,2))
#Simulate a random normal distribution of 20 elements, whose mean is 5 and standard deviation 2.5 . 
#Capture the result in x5.
x5 = 5 + 2.5*np.random.randn(20) # normal distribution with mean 10 and sd 2
x6 = np.arange(0,40,2)
x7 = np.linspace(10,20,30)
x = np.arange(6).reshape(2,3)
y = np.hsplit(x,(2,))
print(y[0])

x = np.arange(20).reshape(10, 10)

import numpy as np

y = np.array([3+4j, 0.4+7.8j])
print(y.dtype)
type(y.flags)

x = np.array([[3.2, 7.8, 9.2],
             [4.5, 9.1, 1.2]], dtype='int64')
print(x.itemsize)

n = [5, 10, 15, 20, 25]
x = np.array(n)
type(x)
x.size
x.ndim
x.shape

n = [[-1, -2, -3, -4], [-2,-4, -6, -8]]
y = np.array(n)
y.ndim
y.shape
y.size
y.dtype
y.nbytes

x = np.arange(3, 15, 2.5) # 2.5 is step
print(x)
y = np.linspace(3, 15, 5) # 5 is size of array 'y'
print(y)
x = np.random.rand(2) # 2 random numbers between 0 and 1
print(x)
x = np.random.randn(3) # Standard normal distribution
print(x)
x = 10 + 2.5*np.random.randn(3) # normal distribution with mean 10 and sd 2
print(x)

z = np.eye(2)
print(z)
print(np.array(([1, 2], (3,4))).shape)

 aa = [[[-1,1],[-2,2]],[[-3 3], [-4, 4]]]
aa.ndim
aa.size
aa.shape

x = np.arange(30).reshape(6, 5)
res = np.vsplit(x, (2, 5))
print(res)

x = np.arange(20).reshape(10, 10)

x = np.arange(4).reshape(2,2)
y = np.vsplit(x,2)
print(y[0])

x = np.arange(6).reshape(2,3)
y = np.hsplit(x,(2,))
print(y[0])

zz  = np.arange(1,21)
zz.shape
y = zz.reshape(2,10)
ab = np.hsplit(y,2)

aab = zz.reshape(4,5)
np.vsplit(aab,2)

aa = np.array([3,6,9,12])
p = aa.reshape(2,2)

arr = np.array([15, 18, 21, 24, 27, 30])
q = arr.reshape(2,3)
np.hstack((p,q))
print(np.repeat(3, 4))

x = np.arange(20).reshape(4,5)
print(x.mean(axis=1))

x1 = np.arange(30).reshape(5,6)
print(x1.argmax(axis=1))

x2 = np.array([[-2],[2]])
y2 = np.array([[-3, 3]])
print(x2.dot(y2))


x3 = np.array([[0, 1], [1, 1], [2, 2]])
y3 = x3.sum(-1)
print(x3[y3 < 2, :])

x6 = np.arange(4)
print(x6.flatten())

x4 = np.arange(30).reshape(3,5,2)
print(x4[-1, 2:-1, -1])


x8 = np.arange(30).reshape(3,5,2)
print(x8[1,::2,1])

x6 = np.arange(30).reshape(3,5,2)
print(x6[1][::2][1])
x6 = np.arange(30).reshape(3,5,2)
print(x6[-1, 2:-1, -1])
zp  = np.arange(1,31)
r = zp.reshape(6,5)
r[-1]
r[:,2]
r[0:2,2:]

x9 = np.arange(30).reshape(2,3,5)
b = np.array([True,False])
x9[b]
x9[b,:,1:3]

print(np.linspace(1, 10, 5))
print(np.arange(1, 10, 5))

x0 = np.arange(4).reshape(2,2)
y0 = np.vsplit(x0,2)
print(y0[0])

x88 = np.arange(12).reshape(3,4)
print(x88[:,1])

x23 = np.arange(30).reshape(3,5,2)
print(x23[1][::2][1])

x = np.arange(30).reshape(3,5,2)
print(x[-1, 2:-1, -1])

x98 = np.arange(12).reshape(3,4)
print(x98[-1:,].shape)

x78 = np.arange(30).reshape(5,6)
print(x78.argmax(axis=1))


x = np.array([[0, 1], [1, 1], [2, 2]])
y = x.sum(-1)
print(x[y < 2, :])

x = np.arange(6).reshape(2,3)
y = np.hsplit(x,(2,))
print(y[0])

print(np.array(([1, 2], (3,4))).shape)

x = np.array([[-2], 
              [2]])
y = np.array([[-3, 3]])
print(x + y)


x = np.arange(4).reshape(2,2)
y = np.arange(4, 8).reshape(2,2)

print(np.hstack((x,y)))

x = np.arange(20).reshape(4, 5)
x.shape

np.random.rand(3, 2)
np.random.random(3, 2)


x = np.arange(12).reshape(3,4)
print(x[-2])