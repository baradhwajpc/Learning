# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:06:48 2018

@author: baradhwaj
"""

### Py Interview Question
import re

strIp = 'Email_Address,Nickname,Group_Status,Join_Year \
aa@aaa.com,aa,Owner,2014 \
bb@bbb.com,bb,Member,2015 \
cc@ccc.com,cc,Member,2017 \
dd@ddd.com,dd,Member,2016 \
ee@eee.com,ee,Member,2020'
bool isValid
#Get the Domain from this string 

for (i in  re.finditer( '([a-zA-Z]+)@([a-zA-Z]+).(com)' , strIp):
    print i.group(2)
    
    
nameAppend = []
import pandas as pd
import os
os.chdir('G:\Python\Learning\Self Learning\Py Interview Ques')
dfIp = pd.read_csv('marks.csv')

for i in dfIp.iterrows():
    re.finditer('i$|ie$(,)',i)
    nameAppend.append(i.group(1))
print(nameAppend)

a = [1,2,3,4,5]
b = [6,7,8,9]
print(a+b)




N = int(input())
list1 =[]
for i in range(N) :
        cmd = input().strip().split("\t")
        print(cmd)
        if(cmd[0] =='insert'):
            list1.insert(cmd[1])
            list1.insert(cmd[2])
        elif(cmd[0] =='remove'):
            list1.remove(cmd[1])
        elif(cmd[0] =='append'):
            list1.append(cmd[1])
        elif(cmd[0]=='sort'):
            list1.sort()
        elif(cmd[0]=='pop'):
            list1.pop()
        elif(cmd[0]=='print'):
            print(list1)
    
