# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:13:31 2017

@author: baradhwaj
"""

def print_with_exclamation(word):
    print(word+"!!")
#print_with_exclamaion('Hello')
def sumIntVals(x,y):
    print(x+y)
def sumDefault(x=0,y=0):
    print(x+y)
        
    
#SCOPE OF A FUNCTION
def print_val(v):
    v = v+1
print(v)
print_val(10)
a = 10
print_val(a)
#throws error as v is in the scope of function
#not accessible outside the function
#a variable created in function dies at the end of function
    print(v)
#Avoid accessing variables in the global environment inside a function
some_list = [3,1,7,5,10]
def impure_function(ip1):
    some_list.append(ip1)
impure_function(20)
print(some_list)
impure_function(30)
print(some_list)
some_list = [3,1,7,5,10]
def pure_function(l1,ip1):
    l1.append(ip1)
    return l1
some_list = pure_function(some_list,20)
print(some_list)

def calculate(x,y,options):
    if(options=="sum"):
        return (x+y)
    elif(options =="diff"):
        return (x-y)
    elif(options=="mult"):
        return (x*y)
    elif(options=="div"):
        return (x/y)

op1 = calculate(5,3,"sum") # 8
op2 = calculate(5,3,"diff") # 2
op4 = calculate(5,3,"mult") # 15
op6 = calculate(6,3,"div") #2

print(op1)
print(op2)
print(op4)
print(op6)


#ARGUMENT MATCHING
calculate(y = 3, x = 5, options = "diff")
calculate(options = "diff", y = 3, x = 5)
#throws error. positional matching cannot follow argument matching
calculate(options = "diff", 3, 5)
#argument matching can follow position matching
calculate(3, 5, options = "diff")


#One line compact functions
def polynomial(x):
    return x*2 + 5*x + 5
polynomial(5)
polynomial(11)
polynomial_lambda = lambda x:x*2+5*x + 5
polynomial_lambda(5)
polynomial_lambda(11)
#More than one input in lambda
polynomial_lambda2 = lambda x,y:x*2+5*y + 5
polynomial_lambda2(5,10)
polynomial_lambda2(2,45)
#FUNCTIONAL PROGRAMMING

#Passing function as an argument to another function
def add_five(x):
    return (x+5)
add_five(10)

def add_ten(x):
    return (x+10)
    
def call_function(func,x):
    return func(x)
    
call_function(add_five,10)

def apply_nTimes(func,x):
    return func(func(func(func(func(x)))))

def apply_(func,x):
    return(func(func(x)))

    
apply_twice(add_five,10)
apply_twice(add_ten,10)
apply_nTimes(add_ten,10)




