# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 07:30:16 2017
@author: baradhwaj
"""
################################
#Create list  from 1 to 20 -a
listOne = list(range(1,21))
print(listOne)
# create a list from 20 to 1 - b
listTwo = list(range(20,0,-1))
print(listTwo)
# add two list - c
listFour = list(range(1,21))
listFive = list(range(19,0,-1))
listThree = listFour+listFive
print(listThree)
################################
#Print recurent numbers
#Create Temp variable - d
tmp = [4,6,3]
# Create variable with 4,6,3 with 10 occurances of 4
listSix = tmp* 10
print(listSix)
# or
listSeven = [4]*9+tmp
print(listSeven)
# Print 11 4 10 6 and 10 3 
listEight = listSix+[4]
print(listEight)
 #10 4 20 6 and  30 3 
l4 = [4]*10
l6 = [6]*20
l3 =  [3]*30
listNine = l4+l6+l3
print(listNine)
################################