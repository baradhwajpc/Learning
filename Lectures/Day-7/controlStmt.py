# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 10:45:29 2017

@author: baradhwaj
"""

################################# if #################
#else if is called elif
x =10
if x == 4 :
    print ('\n 4')
elif x ==5:
    print('\n 5')
else:
    print('\n 10')
    
## multiple conditions 
x =97
x = 9961
if (x >=5 and x <= 100):
    print ('\n in Range')
else:
    print('\n Not in range') 

## nested if ####
if 9> 7:
    print('\n passed')
    if 5 >9:
        print('\n fail')
########################################################
# for loops
a = [2,6,1,8,9]
for i in a:
    print(i**2)

# avoid    
a_squared= []
for i in a:
    a_squared.append(i**2)
print(a_squared)

## extending a list in loop involves dynamic mem alloc
## avoid doing it
# try this
a_squared= []
a_squared = [0.0]*len(a)
pos = 0
for i in a: # loop thru values and maintain seperate variables for position
    a_squared[pos]  = i**2
    pos = pos+1
print(a_squared)

## loop thru pos use it for both slicing and assigning
a_squared= []
a_squared = [0.0]*len(a)
pos = 0
for i in range(len(a)): # loop thru values and maintain seperate variables for position
    a_squared[i]  = a[i]**2
print(a_squared)


## Ennumaration
# Return a tuple with position and value
for i in enumerate(a):
    print(i)

a_squared= []
a_squared = [0.0]*len(a)
pos = 0
for i in enumerate(a):
    val = i[1]
    pos = i[0]
    a_squared[pos]  = val **2
print(a_squared)


## List comrehension . Elegent looping
a_squared = [i **2 for i in a]
### Vectorized operation or numpy operation
a_squared = np.power(a,2)

### Extracting od nos 
rnd_no = [13,12,44,65,39,28,80]
odd_numbers = []
for i in rnd_no:
    if i% 2 ==1:
        odd_numbers.append(i)
print(odd_numbers)

# list comprhension
odd_numbers = [i for i in rnd_no if i %2 ==1]
print(odd_numbers)
#numpy array
odd_numbers = np.array(rnd_no)
odd_numbers = odd_numbers[odd_numbers%2 ==1]
print(odd_numbers)

## Extract values above 40
listEl = [13,12,44,65,39,28,80]
# for loop
greater40 = []
pos = 0
for i in listEl:
    if i > 40:
        greater40.append(i)
print(greater40)
# list comprehension
print([i for i in listEl if i > 40])
# vectorized op
greater40 = np.array(listEl)
greater40 = greater40[greater40 > 40]
print(greater40)


###### Nested loop ######
arr1 = np.random.randint(10,100,12)
mat1 = arr1.reshape(3,4)

arr2 = np.zeros(len(arr1))
div_by_2 = arr2.reshape(3,4)

nrow = mat1.shape[0]
ncol = mat1.shape[1]

for i in range(nrow) :
    for j in range(ncol):
        div_by_2[i,j] = mat1[i,j]/2

## vectorized operation
mat_div_22 = mat1/2


### While loop###### - Runs till condition fails
x = 2 
while x<20:
    print(x**2)
    x=x+1

## Infinite loop
while True:
    print('1')

### Opptimization problems are generally writte in while loop all have break condition
x = 2 
iter_count = 1
max_iter = 10
while x<20:
    print(x**2)
    x=x-1 ## operation not cenverging
    iter_count = iter_count +1
    if iter_count >= max_iter:
        break # terminates exex
