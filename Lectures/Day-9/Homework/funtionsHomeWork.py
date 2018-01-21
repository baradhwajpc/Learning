# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:55:35 2017

@author: baradhwaj
"""
import pandas as pd
import numpy as np

# Q1 . Series compute difference  between mean and median
randSeries = pd.Series(np.random.randint(50,size=10))
print(randSeries)
def meanMedianDiff(inputSeries):
    return(np.mean(inputSeries)-np.median(inputSeries))
meanMedianDiff = meanMedianDiff(randSeries)
print(meanMedianDiff)

# Q2.
def max_var2_corresponding(df,col1,col2):
    return df[col2][df[col1] == np.max(df[col1])].values[0]
              
'''Test Case 1:
Create a dataframe score_df using following set of commands
Call the function developed by you with following statements. Expected
outcome is provided after #
'''
#random_df.loc[:,["A","C"]]
#random_df.iloc[:,[0,2]]
# Input:
math_score_array = np.array([95,67,88,45,84])
eng_score_array = np.array([78,67,45,39,67])
gender_array = np.array(["M","M","F","M","F"])
score_df = pd.DataFrame({
'Maths':math_score_array,
'English':eng_score_array,
'Gender':gender_array})
score_df.index = ["R1001","R1002","R1003","R1004","R1005"]
outEng = max_var2_corresponding(score_df,'Maths','English')
print(outEng)
outGender = max_var2_corresponding(score_df,'English','Gender')
print(outGender)
# TC2: Create a dataframe emp_details using following set of commands
emp_details_dict = {
'Age': [25,32,28],
'Income': [1000,1600,1400]
}
emp_details = pd.DataFrame(emp_details_dict)
emp_details.index = ['Ram','Raj','Ravi']
#Call the function developed by you with following statements. Expected
#outcome is provided after #
# max_var2_corresponding(emp_details,"Income","Age") #32
print(max_var2_corresponding(emp_details,"Income","Age"))