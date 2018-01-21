# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:24:16 2017

@author: baradhwaj
"""

import pandas as pd
import numpy as np
import os

acs_2013_data = pd.read_csv("G:\\Python-Learning\\Lectures\Day-5\\Data\ACS_13_5YR_S1903.csv")
acs_2008_data = pd.read_csv("G:\\Python-Learning\\Lectures\Day-5\\Data\\ACS_08_3YR_S1903.csv")
# get current working dir
os.getcwd()
os.chdir("G:\\Python-Learning")
os.getcwd()
acs_2013_data = pd.read_csv("Lectures\\Day-5\\Data\ACS_13_5YR_S1903.csv")
acs_2008_data = pd.read_csv("Lectures\\Day-5\\Data\\ACS_08_3YR_S1903.csv")

## slice forst 7 columns
#1
first6Cols = acs_2013_data.iloc[:,0:7]
#2
first6Cols.columns = ["ID","FIPS","State","Household","Total Household MOE", "Income","Income MOE"]
#3
avgUSASal = np.mean(first6Cols["Income"])
print(avgUSASal)
# 4
maxIncome = np.max(first6Cols["Income"])
cond1 = first6Cols["Income"] == maxIncome
maxState = first6Cols.loc[cond1,"State"]
# 5
minIncome = np.min(first6Cols["Income"])
cond2 = first6Cols["Income"] == minIncome
minState = first6Cols.loc[cond2,"State"]

#6 - 
cond3 = first6Cols["Income"] > avgUSASal
greatAvg = first6Cols.loc[cond3,"State"]

#7 - Texas Income
cond4 =   first6Cols["State"] == "Texas"
texasIncome = first6Cols.loc[cond4,"Income"]
# 8 .2 highest
secHigh = np.sort(first6Cols["Income"])
cond5 = first6Cols["Income"] == secHigh[-2]
stateSecond =  first6Cols.loc[cond5,"State"] 

first6ColsSorted = first6Cols.sort_values("Income",ascending=False)
first6ColsSorted.iloc[1,2]


## Pandas while detecting data types can do mistake
acs_2013_data.dtypes
acs_2013_data = pd.read_csv("Lectures\\Day-5\\Data\ACS_13_5YR_S1903.csv",dtype={"GEO.id2":str})
acs_2008_data.dtypes
acs_2008_data = pd.read_csv("Lectures\\Day-5\\Data\ACS_08_3YR_S1903.csv",dtype={"GEO.id2":str})



################### Data merging- Join in sql ###

#############################
income_2013 = acs_2013_data.iloc[:,[1,3,5]]
income_2013.columns =  ["FIPS","Total Household_2013","Income_2013"]
income_2008 = acs_2008_data.iloc[:,[1,3,5]]
income_2008.columns =  ["FIPS","Total Household_2008","Income_2008"]


### if ref col name is same
merged_income = pd.merge(income_2008,income_2013,on="FIPS")

##  if ref are different
income_2013 = acs_2013_data.iloc[:,[1,3,5]]
income_2013.columns =  ["FIPS_2013","Total Household_2013","Income_2013"]
income_2008 = acs_2008_data.iloc[:,[1,3,5]]
income_2008.columns =  ["FIPS_2008","Total Household_2008","Income_2008"]
merged_income = pd.merge(income_2008,income_2013,left_on="FIPS_2008",right_on="FIPS_2013")
### add new values in 2 datasets
income_2008.loc[52,:] = ["88",12345,123]
income_2013.loc[52,:] = ["99",54321,321]
### inner join 
merged_income_inner = pd.merge(income_2008,income_2013,how="inner",
                               left_on="FIPS_2008",right_on="FIPS_2013")
merged_income_outer = pd.merge(income_2008,income_2013,how="outer",
                               left_on="FIPS_2008",right_on="FIPS_2013")

merged_income_left = pd.merge(income_2008,income_2013,how="left",
                               left_on="FIPS_2008",right_on="FIPS_2013")
merged_income_right = pd.merge(income_2008,income_2013,how="right",
                               left_on="FIPS_2008",right_on="FIPS_2013")
# convert ex to int variable type conversion
merged_income_right["Total Household_2013"]  = merged_income_right["Total Household_2013"].astype(int)

import datetime

### file write
merged_income.to_csv("Lectures\\Day-5\\output/merged_income.csv")
## Without index
now = datetime.datetime.now().date()
fname = "Lectures/Day-5/output/"+str(now)+"_merged_income.csv"
merged_income.to_csv(fname,index = False)