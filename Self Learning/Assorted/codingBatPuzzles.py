# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:37:45 2017

@author: baradhwaj
"""
#Sourced from : http://codingbat.com/python

#Q1:The parameter weekday is True if it is a weekday, and the parameter vacation is True
# if we are on vacation. We sleep in if it is not a weekday or we're on vacation. Return True if we sleep in.
'''
sleep_in(False, False) → True
sleep_in(True, False) → False
sleep_in(False, True) → True
'''
def sleep_in(weekday, vacation):
   if (weekday is False  and vacation is False):
      return True
   elif (weekday is True  and vacation is False):
      return False
   elif (weekday is False  and vacation is True):
      return True
   elif (weekday is True  and vacation is True):
      return True  

#Q2. We have two monkeys, a and b, and the parameters a_smile and b_smile indicate if each is smiling. We are in trouble if they are both smiling or if neither of them is smiling. Return True if we are in trouble.
'''
monkey_trouble(True, True) → True
monkey_trouble(False, False) → True
monkey_trouble(True, False) → False
'''
def monkey_trouble(a_smile, b_smile):
  if (a_smile is False  and b_smile is False):
      return True
  elif (a_smile is True  and b_smile is False):
      return False
  elif (a_smile is False  and b_smile is True):
      return False
  elif (a_smile is True  and b_smile is True):
      return True  
     

#Q3. Given two int values, return their sum. Unless the two values are the same, then return double their sum.
'''
sum_double(1, 2) → 3
sum_double(3, 2) → 5
sum_double(2, 2) → 8
'''
def sum_double(a, b):
   if(a==b):
       return 2*(a+b)
   else : 
       return a+b
#Q3. Given an int n, return the absolute difference between n and 21, 
#except return double the absolute difference if n is over 21.
'''
diff21(19) → 2
diff21(10) → 11
diff21(21) → 0 
'''
def diff21(n):
    if(21 > n):
       return 21-n
    else : 
       return 2*(n-21)

#Q4.We have a loud talking parrot. The "hour" parameter is the current hour time in the range 0..23.
# We are in trouble if the parrot is talking and the hour is before 7 or after 20. Return True if we are in trouble.
'''
parrot_trouble(True, 6) → True
parrot_trouble(True, 7) → False
parrot_trouble(False, 6) → False
'''
def parrot_trouble(talking, hour):
    cond = 7 <= hour <= 20 
    if (talking is True  and  cond is True):
      return False
    elif (talking is True  and  cond is False):
      return True
    else:
      return False
print(parrot_trouble(True,23))

#Q5. Given 2 ints, a and b, return True if one if them is 10 or if their sum is 10.
'''
makes10(9, 10) → True
makes10(9, 9) → False
makes10(1, 9) → True
'''
def makes10(a, b):
    a+b ==10
    a ==10
    b==10
    if((a+b==10) or ((a==10)or(b==10))):
        return True
    else:
        return False
# Q6.Given an int n, return True if it is within 10 of 100 or 200. .
#Note: abs(num) computes the absolute value of a number.
'''
near_hundred(93) → True
near_hundred(90) → True
near_hundred(89) → False      
'''
def near_hundred(n):
    if ((abs(100-n) <=10) or (abs(200-n) <=10)):
        return True
    else:
        return False