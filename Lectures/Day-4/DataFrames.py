# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:13:07 2017

@author: baradhwaj
"""
## Slice by position and index
import pandas as pd
math_score_series = pd.Series([[95,44],55,88,45,69],index =['a','b','c','d','e'])

math_score_series[2]
math_score_series['c']


###########################Panda Data Frame#############################

############ Most used data type in the community
### Saves data as table ####################################

####  2 ways of creating dataframe : from dict and from matrix###
### WWhen importing external files it is automatically saved to data frame####
dict1 = {'col1':pd.Series([1,2,3]),
         'col2' :pd.Series([2,3,4])}
df_dict = pd.DataFrame(dict1)

dict2 = {'col1' :pd.Series([1,2,3],index=['a','b','c']),
         'col2' :pd.Series([2,3,4],index=['a','b','d'])}
df_dict2 = pd.DataFrame(dict2)

dict3 = {'col1':pd.Series([1,2,3],index=['a','b','c']),
         'col2' :pd.Series([2,3,4],index=['a','b','c'])}
df_dict3 = pd.DataFrame(dict3)


dict4 = {'col1':[1,2,3],
         'col2' :[2,3,4]}
df_dict4 = pd.DataFrame(dict4)
df_dict4.index = ['a','b','c']


 # df from dictionary
 math_score_array = np.array([98,47,77,69,88])
 english_score_array = np.array([98,47,77,69,88])
 gender_array = np.array(['m','f','m','f','m'])
 rno_array = ["R1","R2","R3","R4","R5"]
 score_dict = {
                
                'Maths' : math_score_array,
               'English':english_score_array,
               'Gender' : gender_array
               }
score_df = pd.DataFrame(score_dict)
score_df.index = rno_array
 # df from numpy matrix
 
 ### Copy from file 
 
 ################
 
 #########Attributes of Dataframe ###########3
 score_df.index
 score_df.columns
 score_df.shape  # returns tuple with no of rows and columns
 score_df.shape[0] # no of rows
 score_df.shape[1] # no of cols
 ############ Slice dataframe
 
score_df["Maths"] # slicing by column name
math_score_sliced = score_df["Maths"]
type(math_score_sliced)
## slicing multiple columns
cols = ["Maths", "English"]
selected_columns = score_df[cols]
type(selected_columns)

## Slice a row
#score_df["R4"]

## Slice like a matrix in row, column format
## Slice using index and column
score_df.loc["R4",:"Maths"]
score_df.loc["R4",:]
score_df.loc[:,"Maths"]
roi = ["R1","R4"]
coi = ["Maths","Gender"]
score_df.loc[roi,coi]

#### Slice by position
score_df.iloc[3,2]
score_df.iloc[3,:]
score_df.iloc[:,2]
roi = [3,1]
coi = [0,2]
score_df.iloc[roi,coi]
score_df.iloc[0:4,:] # slice range of rows


######## Conditional Boolean slicing #####
# .loc for boolean slicing
# Get math mark of male student
con1 = score_df['Gender']=='m'
score_df.loc[con1,"Maths"]
score_df.loc[con1,["Maths","Gender"]]
score_df.loc[con1,:]

###  above 60 in eng - math score
cond2 = score_df["English"]>60
mat_score = score_df.loc[cond2,"Maths"]
np.mean(eng_score)
avg_eng = sum(eng_score) /len(eng_score) 
cond3 = average(score_df["Maths"][cond2])
score_df.loc[cond3,:]


