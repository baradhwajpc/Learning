# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 09:25:25 2017

@author: baradhwaj
"""
####integer######
# Num without decimal is automatically assigned as integer
a = 2
type(a)
c = -10
type(c)
####float########
# Num with decimals is automatically assigned to float
b = 2134.3423
type(b)


###### String ################
# any val within qutes is string
s= "god"
int(s) # throws error as god could not be converted as an integer
type(s)
s2='doood' # single quote can be used fpr string
type(s2)
s3 = "444" # numbers inside "" is considered as string
int(s3)+5
#int(s) -- throws error . Cannot convert string to int
# s3+ 5  throws error
s4="g" # single char is string
type(s4)

str1= "I"
str2="myself"
concat_str = str1+"me"+str2
print(concat_str)
len(str1)# no of char in integer
len(str2)
 
######################## Boolean ########################3

f = True
type(f)
#g=true # python is case sensitive
h= True
H= False
# output of conditions are boolean
5 > 6
6 > 5
5 == 6 # == for equality check
5 == 5
5 != 5
6 != 5
not(6 == 5)
'god' == 'god' # true
'god'=='odg' # true
'god'== 'God' # false
"god" == 'god' # Quote comparision

# Conditional operatirs - And , or operators
(6>5) and (8>7)
(6>5) and (7>8)
(6>5) and (7>8)
not(6==6) and (8>6)
not((9==6) and (8>7))

############# complex #######################
h = 5+10j
type(h)
m=5-10j
type(m)
h*m
abs(h*m)

###### type casting###########33
float(3)
int(10.5)
# not needed
int(49845029842390482309482309482390482390482309423840923842390482390482353453455555535555533334444477)
   # 4534534534534534534535)
float(4984502984239048230948230948239048239048230942384092384239048239048235345345555553555553333444447744444.534534535353453453453453453453453445)
#end of not needed

## tuple
tup1 = (1,2,3,4,5)
type(tup1)
tup2=("3eq","232","232334","@4242423","423423")
tup3=(6+10j,232,"ewer232334",4242423,"4234234324eqweqw3")

## Ty[le slicing by position
# position starts with 0
# [] square braket for slicing
tup1[0]
tup1[2] = 100 # cant remove. Once created its readonly

#################3# Lists#########################
lst1 = [3,4,5,6,5]
lst1[2] = "100" # supports update and composite data type list
# concat 2 list
lst3=(6+10j,232,"ewer232334",323,3223,233234242423,"4234234324eqweqw3")
lst4 =lst1+lst3

# append a value
lst1.append(304)
print(lst1)

# data simulation
lst5 = list(range(100))
len(lst4)
lst7 = list(range(1,101)) # Generate list of items from 1 to 101
lst5 = list(range(1,101,2)) # Generte list of items starting from 1 to 101 with incrementor set to 2
lst6 = list(range(101,1,-2)) # Generte list of items starting from 1 to 101 with decrementor set to 2

# print 5 's 100 times # Print a number n times
five_rep_100 = [5]* 100
print(five_rep_100)
one_to_five_rep = list(range(1,6))*20
print(one_to_five_rep)

 
## list slicing vs deleting
# slicing  copies the dliced variable and saves in another location
# delete removes physically from memory

# list slicing
l1 = list(range(1,101))
l1[0] # slices forst element
l1[len(l1)-1] # slices out last element
l1[-1] # negative index
l1[-2] # last but 1
l1[5:11] # five to 10 pos
l1[5:] # 5 to last 
l1[:6] # 0 to 5 pos
l1[1:-1] # first to last 
l1[0:2]+l1[-2:] # beg 2 and ending 2
l1[::2] # extracting even pos
l1[1::2] # extracting odd pos

# list searcing
l10 = [10,55,14,147,157,16684,547]
10 in l10
147 in l10
word_list = ['sdf','er','rrt']
'sdf' in word_list

# del operation



