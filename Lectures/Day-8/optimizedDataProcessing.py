# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 08:57:25 2017

@author: baradhwaj
"""
import numpy as np
import pandas as pd

df = pd.DataFrame({
'A':['foo','bar','foo','bar','foo','bar','foo','foo'],
'B':['one','two','two','one','one','two','two','one'],
'C':np.random.rand(8),
'D':np.random.rand(8),
'E':np.random.rand(8),
'F':np.random.rand(8),
'G':np.random.rand(8)})

# For loop for each column in c,d,e,f and g
coi = df.columns[2:]
avg_cols = pd.Series([0.0]*len(coi),index = coi)
for i in coi:
    avg_cols[i] = np.mean(df[i])
print(avg_cols)

nrows = df.shape[0]
avg_rows = pd.Series([0.0]*nrows)
for i in range(nrows):
    avg_rows[i] = np.mean(df.iloc[i,2:])
print(avg_rows)


############### apply approach ###################
df.iloc[:,2:].apply(np.mean,axis=0)# column mean
df.iloc[:,2:].apply(np.mean,axis=1)# row mean

# median
df.iloc[:,2:].apply(np.median,axis=0)# column median
df.iloc[:,2:].apply(np.median,axis=1)# row median


df_foo = df.loc[df["A"]=="foo",:]
df_bar = df.loc[df["A"]=="bar",:]
df_grouped_A = df.groupby("A")
df_foo = df_grouped_A.get_group("foo")
df_bar = df_grouped_A.get_group("bar")
for i in df_grouped_A:
    print (i)
    
# Group by multiple columns unique combo of values in A and B
df_grouped_AB = df.groupby(["A",'B'])
for i in df_grouped_AB:
    print (i)
    
### Group by and apply,aggregate
#repeating a function on a column for each group/segment
df_grouped_A["C"].apply(np.mean)
df_grouped_A["D"].apply(np.mean)
df_grouped_A["G"].apply(np.mean)
#repeating a function on multiple columns of each group
df_grouped_A[["C","D","E","F","G"]].apply(np.mean)


#### Aggregate function ######

#repeating a function on multiple columns of each group
df_grouped_A[["C","D","E","F","G"]].agg(np.mean)
df_grouped_A[["C","D","E","F","G"]].agg(np.median)
df_grouped_A[["C","D","E","F","G"]].agg(np.sum)
#repeating multiple functions on multiple columns of each group
df_grouped_A[["C","D","E","F","G"]].agg([np.mean,np.median])

df_grouped_A[["C","D","E"]].agg({
"C":np.mean,
"D":np.median,
"E":[np.min,np.maxxdee


