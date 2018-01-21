# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 06:53:09 2017

@author: baradhwaj
"""
#### Drug Usage #####

# import 
import os
import numpy as np
import pandas as pd
import matplotlib as plt
# read and print initial data load
os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Drug Addiction')
drugUsage = pd.read_csv('addictionData.csv')
print(drugUsage)
# Manually introduced 2 new columns avgAge and legal
# Fill empty values in all columns
print(np.mean(drugUsage['cocaine-frequency']))
drugUsage['cocaine-frequency'].fillna(np.mean(drugUsage['cocaine-frequency']))
# Datas with hypens should be removed and replaced
print(np.mean(drugUsage['meth-frequency']))
drugUsage['meth-frequency'].fillna(np.mean(drugUsage['meth-frequency']))

print(np.mean(drugUsage['crack-frequency']))
drugUsage['crack-frequency'].fillna(np.mean(drugUsage['crack-frequency']))

print(np.mean(drugUsage['heroin-frequency']))
drugUsage['heroin-frequency'].fillna(np.mean(drugUsage['heroin-frequency']))

print(np.mean(drugUsage['inhalant-frequency']))
drugUsage['inhalant-frequency'].fillna(np.mean(drugUsage['inhalant-frequency']))

print(np.mean(drugUsage['oxycontin-frequency']))
drugUsage['oxycontin-frequency'].fillna(np.mean(drugUsage['oxycontin-frequency']))
# End of cleanup
# Which age group has maximum sample data
print(drugUsage.loc[drugUsage['n'] ==np.max(drugUsage['n']) , ['age']])
# Age grooup with max alchol use
print(drugUsage.loc[drugUsage['alcohol-use'] ==np.max(drugUsage['alcohol-use']) , ['age']])

# Age  with alcohol use above average alcohol use  and tranquilizer-use below tranquilizer-use
cond = drugUsage['alcohol-use'] > np.mean(drugUsage['alcohol-use']) 
cond1 = drugUsage['tranquilizer-use'] < np.mean(drugUsage['tranquilizer-use'])
print(drugUsage.loc[cond & cond1,'age'])

## Less meth use and more inhalat use
cond = drugUsage['meth-use'] < np.mean(drugUsage['meth-use']) 
cond1 = drugUsage['inhalant-use'] > np.mean(drugUsage['inhalant-use'])
print(drugUsage.loc[cond & cond1,'age'])

## Less alcohol and more sedatives
cond = drugUsage['alcohol-use'] < np.mean(drugUsage['alcohol-use']) 
cond1 = drugUsage['sedative-use'] > np.mean(drugUsage['sedative-use'])
print(drugUsage.loc[cond & cond1,'age'])

## Tried out random graphs
# inhalant use by age
drugUsage['inhalant-use'].plot.box(by=['age']) # pandas
# drugUsage['ageAvg'].plot.box(by=['alcohol-use','heroin-use']) # pandas
drugUsage['ageAvg'].plot.box(by=['alcohol-use']) # pandas

# cocain and tranquilizer use based on legality
pl = drugUsage.boxplot(column='cocaine-use',by='legal') # matlablib
pl = drugUsage.boxplot(column='tranquilizer-use',by='legal')

# Show the relation between alcohol use and age
scatterPlot = drugUsage.plot.scatter('alcohol-use','ageAvg')
scatterPlot.set_xlabel('Alcohol Use')
scatterPlot.set_ylabel('Age')

# Indivudual Scatters
#plt.scatter(drugUsage.ageAvg,drugUsage['hallucinogen-use'])
#plt.scatter(drugUsage.ageAvg,drugUsage['alcohol-use'])
#plt.scatter(drugUsage.ageAvg,drugUsage['tranquilizer-use'])
#plt.scatter(drugUsage.ageAvg,drugUsage['meth-use'])
#plt.scatter(drugUsage.ageAvg,drugUsage['sedative-use'])
# Combined Picture Attempt 1 
ax1 = drugUsage.plot(kind='scatter', x='ageAvg', y='hallucinogen-use', color='r' , label ="age vs hallu use")    
ax2 = drugUsage.plot(kind='scatter', x='ageAvg', y='alcohol-use', color='g', ax=ax1 , label = "age vs alcohol use")    
ax3 = drugUsage.plot(kind='scatter', x='ageAvg', y='meth-use', color='b', ax=ax1 , label = "age vs meth use")
ax1.set_xlabel("age")
ax1.set_ylabel("All addictions")
print(ax1==ax2==ax3)

# Attempt 2  line
ax = drugUsage.plot(kind="line", x="ageAvg",y="alcohol-frequency", color="b",  label ="age vs alcohol use")
drugUsage.plot(kind="line",x="ageAvg",y="hallucinogen-frequency", color="r",  label ="age vs hallu use", ax=ax)
drugUsage.plot(kind="line",x="ageAvg",y="marijuana-frequency", color="g",  label ="age vs marijuana use", ax=ax)
drugUsage.plot(x="ageAvg",y="meth-frequency", color="orange",  label ="age vs meth use", ax=ax)
drugUsage.plot(x="ageAvg",y="pain-releiver-frequency", color="purple",  label ="age vs pain-releiver use", ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("All Addictions")
plt.show()
# Attempt 2 scatter 
ax = drugUsage.plot(kind="scatter", x="ageAvg",y="alcohol-frequency", color="b",  label ="age vs alcohol use")
drugUsage.plot(kind="scatter",x="ageAvg",y="hallucinogen-frequency", color="r",  label ="age vs hallu use", ax=ax)
drugUsage.plot(kind="scatter",x="ageAvg",y="marijuana-frequency", color="g",  label ="age vs marijuana use", ax=ax)
drugUsage.plot(kind="scatter",x="ageAvg",y="meth-frequency", color="orange",  label ="age vs meth use", ax=ax)
drugUsage.plot(kind="scatter",x="ageAvg",y="pain-releiver-frequency", color="purple",  label ="age vs pain-releiver use", ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("All Addictions")

#plt.show()


colnames = list (drugUsage.columns)
drugUsage.reset_index().plot(x="ageAvg", y=colnames[5:13], kind = 'line', legend=False, 
                 subplots = True, sharex = True, figsize = (10.5,10), ls="none", marker="o")