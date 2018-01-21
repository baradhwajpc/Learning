# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:08:35 2017

@author: baradhwaj
"""
#### Create New Dictionary ###########################
dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}

##Access Items with Key######
print('Name: ' , dict['Name'])

### Update item in dict with key ##################3
dict['Age'] = 8; # update existing entry

#########Add item to existng dict
dict['School'] = "DPS School"; # Add new entry
print(dict)

###### Delete Dictionary items##############
del dict['Name'] ## delete based on key
print(dict)
dict.clear() ## flushes the content in dictionary
del dict; # deletes the dictionary

########## Duplicate Keys Behavior###########
dict = {'Name': 'Zara', 'Age': 7, 'Name': 'Manni'} 
#2 names here:manni overrides zara
print ("dict['Name']: ", dict['Name'])

#########Keys are immutable : permitted datatypes : string,int ,tuple#########
dict = {['Name']: 'Zara', 'Age': 7}
print ("dict['Name']: ", dict['Name']) # throws error since key cant be list

##########Built in funcgtionalities of dictionary ###########################
#### Length#####
dict1 = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
dict2 = {'Name': 'Dora', 'Age': 3, 'Class': 'PKG'}
len(dict1)
compare(dict1,dict2)  ######Clarification

#########GET####
dict1.get('Name')
### HAs KEy
dict1.has_key('Age') ### clarification####

####Dict update with another dict
dict3 = {'Gender' : 'F',}
dict1.update(dict3)## update is used to add aattributes that are not in the other dictionary
print(dict1)
print(dict1.keys()) # prints all keys
print(dict1.items()) # prints all items
print(dict1.setdefault('Age',None)) # gets the default value else set none

########## From key : creates new dict with user gicven input
# vowels keys
keys = {'a', 'e', 'i', 'o', 'u' }
vowels = dict.fromkeys(keys)
print(vowels) # Note: values here are null coz we dint set any

# vowels keys
keys = {'a', 'e', 'i', 'o', 'u' }
value = 'vowel'
vowels = dict.fromkeys(keys,value)
print(vowels)

### dictionaries with immutable values ########
# vowels keys
keys = {'a', 'e', 'i', 'o', 'u' }
value = [1]
vowels = dict.fromkeys(keys, value)
print(vowels)
# updating the value
value.append(2) # creates a new dict entry with value appended to 2 ie [1,2],a
print(vowels)

##################Pop################
sales = { 'apple': 2, 'orange': 3, 'grapes': 4 }
element = sales.pop('apple') # returns the item key  poped
print('The popped element is:', element)

print('The dictionary is:', sales)
#If key is not found - value specified as the second argument (default)
les = { 'apple': 2, 'orange': 3, 'grapes': 4 }
element = sales.pop('kiwi','apple')
print('The popped element is:', element)
print('The dictionary is:', sales)
#popitem() returns and removes an arbitrary element(key,value)pair from dict.
person = {'name': 'Phill', 'age': 22, 'salary': 3500.0}
result = person.popitem()
print('person = ',person)
print('Return Value = ',result)
##############Iterating in dictionary################3333
d = {"a":123, "b":34, "c":304, "d":99}
for key in d.keys():
     print(key)
     
#### COnvert list to dictionaries
dishes = ["pizza", "sauerkraut", "paella", "hamburger"]
countries = ["Italy", "Germany", "Spain", "USA"]
foodFromCountry = zip(dishes,countries) 
print(foodFromCountry)## This is an iterator cannot be printed
lst =list(foodFromCountry)
# to Print convert to .list()
print(list(foodFromCountry))
import builtins as bl   ## builtins package needs to be imported for 
dictFfromC = bl.dict(lst)
print(dictFfromC)

### One list has elemts more than the other list. The exceeding element is discarded
dishes = ["pizza", "sauerkraut", "paella", "hamburger",'dosa']
countries = ["Italy", "Germany", "Spain", "USA"]
#foodFromCountry = zip(dishes,countries)
import builtins as bl   ## builtins package needs to be imported for 
dictFfromC = bl.dict(zip(dishes,countries))
print(dictFfromC)
################## End of dictionary ##################################