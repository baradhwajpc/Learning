# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 08:47:45 2017

@author: baradhwaj
"""
############################## list creation ##################################
n_list = ["Happy", [2,0,1,5]]
# Nested indexing
# Output: a
print(n_list[0][1])    
# Output: 5
print(n_list[1][3])

##################Accesing Elements in the list###################
mylist = ['p','c','b','a','r','a','d','h']
print(mylist[2:5]) # prints between 2 and 6
# negative index numbering starts from list end to list beg:  -1 
print(mylist[:-2]) # prints until index -2
print(mylist[1:]) # prints starting from index 1 

##############Changing Elements ################################
# mistake values
odd = [2, 4, 6, 8]
# change the 1st item    
odd[0] = 1            
# Output: [1, 4, 6, 8]
print(odd)
# change 2nd to 4th items
odd[1:4] = [3, 5, 7]  
# Output: [1, 3, 5, 7]
print(odd)    

##################Adding items###################################
#### append ######
odd.append(9)
print(odd)  
###extend: adds multiple items exclosed within []################
odd.extend([11,13,15,17])
print(odd)  

##########Insert item to the list ###############################
odd = [1,9]
odd.insert(1,3)
print(odd)
odd[2:2] = [5,7]
print(odd)
######################delete item using values####################

##############Remove items #################33
animal = ['cat', 'dog', 'rabbit', 'guinea pig']
animal.remove('dog')
print('Updated: ' ,animal)
####Removing duplicates : Removes only the first occurance#######
animal = ['cat', 'dog','dog', 'rabbit','dog', 'guinea pig']
animal.remove('dog')
print('Updated: ' ,animal)

###########Count of an item found in the list########33
print(animal.count('dog'))
#### del one item######
del odd[2]
print(odd)

############# Sorting ##################################
vowels = ['a','e','i','o','u']
print(vowels.sort()) # no paramenters
print(vowels.sort(reverse=True)) # 1 param : reverse = True
print(sorted(vowels,reverse=True)) # Alternative for using sort

########Custom Sort function#########################
# random list

######Del a range######
del odd[1:5]
print(odd)
######################delete items using index####################
###pop()###
#with index as argument,we can delete the particular item in list
odd.pop(1)  # pops 13 and prints 1,15,17
print(odd)
#Without index argument mentioned ,it deletes the last element- FILO Stack DS#
odd.pop() # pops 17 and prints 1,15
print(odd)
#################Deleting the entire list#########################
even = [2, 4, 6, 8]
odd = [1,3,5,7]
del even 
print(even)
################Clearing the list - flushes the value in the list######
odd = [1,3,5,7]
odd.clear()
print(odd)
###################Delete based on slice################################
mylist = ['p','c','b','a','r','a','d','h']
mylist[2:3] = [] # deletes items from 2 to 4 index
print(mylist)

###############################################################################
os = ['Windows', 'macOS', 'Linux']
print('Original List:', os)
# List Reverse
os.reverse()
# updated list
print('Updated List:', os)
reversed_list = os[::-1]
print('Updated List:', reversed_list)
for o in reversed(os):
    print(o)
####################################End of List Learning######################