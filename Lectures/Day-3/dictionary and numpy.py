# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 07:05:53 2017

@author: baradhwaj
"""
# List
math_score_list = [95,67,88,45,84]
type(math_score_list)
print(math_score_list[2])

#Dictionary

# Doubts
# can we have a list as value ? - Yes 
'''what data types are permitted to the key - Immutable Data Types are allowed. 
String , int , float, '''
#  Can have list of dicionaries and dictionaries of list

math_score_dict = {
'Bharadwaj':95,
'Johnson':67,
'Prabhakaran':88,
'Roshan':45,
'Divya':84}
math_score_dict['Johnson']
math_score_dict['Roshan']
# Throws error since kannan is not available
math_score_dict['Kannan'] 
math_score_dict['Kannan'] = 94 
print(math_score_dict)
del math_score_dict['Kannan']
math_score_dict['Roshan'] = 55 # changing value of key

# Excersice

states = {
'Oregon':'OR',
'Florida':'FL',
'California':'CA',
'New York': 'NY',
'Michigan':'MI'}
cities = {
'CA':'San Fransico',
'MI': 'Detroit',
'NY':'Manhattan'}

# add Orlando to FL
cities['FL'] = 'Orlando'
print(cities)
cities['MI']
# Wroong cities['NY']
cities[states['New York']]


##########################numpy array#########################################
# has mean , meadian , sd packages in it
import numpy as np
math_score_list = [95,67,88,45,84]
maths_score_array = np.array(math_score_list)
np.mean(maths_score_array)
np.median(maths_score_array)
np.std(maths_score_array)
# np commands canbe applied to list
np.mean(math_score_list)

# Conditional slicing is esy in numpy array

#slicing based on condition is done thru for loop in list
abv70=[]
for i in math_score_list:
    if i>70 :
        abv70.append(i)
print(abv70)
        
# easy on numpy arr
math_score_list = [95,67,88,45,84]
maths_score_array = np.array(math_score_list)
condn = maths_score_array >70
abv_70 = maths_score_array[condn]

above_70 = maths_score_array[maths_score_array >70]
print(above_70)

# score b/w 70 and 90
condn1 = maths_score_array >70
condn2 = maths_score_array <90
# & for And and | for OR
maths_score_array[condn1 & condn2]
maths_score_array[condn1 | condn2]

#condition on one array can be used to slice another array provided they are of same length
gender = np.array(['M','F','F','M','M'])
condn = gender == "M"
math_score_array = np.array(math_score_list)
math_score_array[condn]
condn1 = gender == "M"
condn2 = math_score_array > 80
math_score_array[condn1 | condn2]
#Vectorized Operation
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
#element wise sum
math_score_array + eng_score_array
#element wise mean
(math_score_array + eng_score_array)/2


###################numpy matrix########################

#reshaping a numpy array
array1 = np.array([10,67,84,56,70,25,93,73])
len(array1)
mat1 = array1.reshape(2,4)
mat2 = array1.reshape(4,2)
mat1.T # transpose
#stacking arrays
#LIST of arrays can be stacked
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
score_mat1 = np.column_stack([math_score_array,eng_score_array])
score_mat2 = np.row_stack([math_score_array,eng_score_array])
#MATRIX sLICING
#matrix[row_position,column_position]
score_mat1[0,0] # 0th row, 0th column
score_mat1[2,1] # 2nd row, 1st column
score_mat1[:,1] # all rows, 1st column
score_mat1[2,:] #2nd row, all columns
score_mat1[1:4,:] #1st till 3rd row, all columns
score_mat1[[1,3,4],:] # 1st, 3rd and 4th row , all columns


##############33pandas######################33
