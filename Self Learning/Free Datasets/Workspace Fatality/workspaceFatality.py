# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:42:58 2017
@author: baradhwaj
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Change dir and read file
os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Workspace Fatality')
workspaceFatality = pd.read_csv("WorkspaceFatalities.csv",thousands =',')
print(workspaceFatality)
# Change headers
workspaceFatality.columns =['State', 'Number of Fatalities','Rate of Fatalities', 'State Rank',
'Number of Injuries/Illnesses','Injuries/Illnesses Rate',
'Penalties Avg','Penalties FY (Rank)',
'Inspectors','Years','State Federal Program']
print(workspaceFatality)
# Fill out missing values with avg value
print(np.mean(workspaceFatality['Number of Injuries/Illnesses']))
workspaceFatality['Number of Injuries/Illnesses'].fillna(np.mean(workspaceFatality['Number of Injuries/Illnesses']))
# Least no of injuries
print(workspaceFatality.loc[workspaceFatality['Number of Injuries/Illnesses'] 
        == np.min(workspaceFatality['Number of Injuries/Illnesses']),
        'State'])
# Average number of Injury   all state
np.mean(workspaceFatality['Number of Injuries/Illnesses'])
# Most no of injuries
print(workspaceFatality.loc[workspaceFatality['Number of Fatalities'] 
        == np.max(workspaceFatality['Number of Fatalities']),
        'State'])
# Average number of Injury   all state
np.mean(workspaceFatality['Number of Fatalities'])
# Top 3 penality paying state by avg
print(workspaceFatality.sort_values(by='Penalties Avg',ascending=False).head(3)['State'])
# Top 3 Least penality paying state by rank
print(workspaceFatality.sort_values(by='Penalties FY (Rank)',ascending=True).head(3)['State'])
# States with top 5 inspectors
print(workspaceFatality.sort_values(by='Inspectors',ascending=False).head(5)['State'])
## Relate Injuries/Illnesses Rate by State Fedaral Program
workspaceFatality.boxplot(column='Injuries/Illnesses Rate',by=['State Federal Program'])
## Relate Number of Injuries/Illnesses by State Fedaral Program
workspaceFatality.boxplot(column='Number of Injuries/Illnesses',by=['State Federal Program'])
## Relate Number of Fatalities by State Fedaral Program
workspaceFatality.boxplot(column='Number of Fatalities',by=['State Federal Program'])
## Relate Penalties FY (Rank) by State Fedaral Program
workspaceFatality.boxplot(column='Penalties FY (Rank)',by=['State Federal Program'])
# Compare Number of Fatalities by State Federal Program
workspaceFatality.groupby(['Number of Fatalities', 'State Federal Program']) \
.size().unstack('State Federal Program').replace(np.NaN, 0).plot(title="Injury Vs Death",figsize=(20,10))
plt.show()