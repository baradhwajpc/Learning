# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:13:42 2018

@author: baradhwaj
"""

import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import defaultdict

os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Ted Talks')
tedTalks = pd.read_csv('tedTalks.csv')

tedTalks['film_date']=pd.to_datetime(tedTalks['film_date'],unit='s')
tedTalks['published_date']=pd.to_datetime(tedTalks['published_date'],unit='s')
tedTalks['Talk_ID'] = range(1, len(tedTalks)+1)
tedTalks.head()

# Ted talks filimed before ted was established
print(tedTalks.loc[(tedTalks['film_date'].dt.year < 1984),['title','main_speaker','film_date']])
# Most speeches
tedTalks['abbr']= tedTalks['main_speaker'].apply(lambda x:x[:3])
f, ax = plt.subplots(figsize=(20, 18))
sns.factorplot(x="abbr", y="views", data=tedTalks.nlargest(15,'views'), kind="bar",size=7)

tedTalks['year'] = tedTalks['film_date'].dt.year
year_df = pd.DataFrame(tedTalks['year'].value_counts().reset_index())
year_df.columns = ['year', 'talks']
f, ax = plt.subplots(figsize=(20, 18))
sns.pointplot(x='year', y='talks', data=year_df)

## Lengthy speakers and thier profession
# Views vs Years
plt.figure(figsize=(20,10))
sns.swarmplot(x="year", y="views", data=tedTalks)

# Year vs Comments
#plt.figure(figsize=(20,20))
sns.factorplot(x="year", y="comments", data=tedTalks , size=10);
# Year vs Duration
plt.figure(figsize=(20,10))
sns.swarmplot(x="year", y="duration", data=tedTalks)

# Year vs Language
plt.figure(figsize=(20,10))
sns.violinplot(x="year", y="languages", data=tedTalks)
# Dunno What this does
plt.figure(figsize=(20,10))
order = tedTalks.std().sort_values().index
sns.lvplot(data=tedTalks, order=order, scale="linear", palette="mako")

## Views vs comments
sns.jointplot(x='views', y='comments', data=tedTalks)

##########################################################################

# Tag Cloud for professions and Tags
# Source : https://www.kaggle.com/mchirico/quick-look-seaborn-wordcloud
from io import StringIO

# Use nltk for valid words
import nltk
import collections as co
from io import StringIO
import warnings # ignore warnings 
warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

si=StringIO()
tedTalks['speaker_occupation'].apply(lambda x: si.write(str(x)))
s=si.getvalue()
si.close()
from wordcloud import WordCloud
# Read the whole text.
text = s
wordcloud = WordCloud(width=800, height=600).generate(text)
# Open a plot of the generated image.
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

##################################
si=StringIO()
tedTalks['tags'].apply(lambda x: si.write(str(x)))
s=si.getvalue()
si.close()
from wordcloud import WordCloud
# Read the whole text.
text = s
wordcloud = WordCloud(width=800, height=600).generate(text)
# Open a plot of the generated image.
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
#############################################################################
### New plots in matplot lib
# Create a pie chart
plt.figure(figsize=(12,12))
tedTalks['year'].value_counts().plot.pie(
shadow=False,
    autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()
plt.show()

#tedTalks['film_year'].value_counts().plt.pie(labels=tedTalks['film_year'],shadow=False,
#explode=(0, 0, 0, 0, 0.15),startangle=90,autopct='%1.1f%%')

#############################################################################
tagsList = []
for index, row in tedTalks['tags'].iteritems():
    for values in [i.split(',', 1)[0] for i in row]:
        if values.endswith(']'):
            values.replace(']','')           
            tagsList.append([values, "true"])            
        elif values.startswith('['):
            values.replace('[','')         
            tagsList.append([values, "true"])
        else:
            tagsList.append([values, "true"])

rating_names = set()
for index, row in tedTalks.iterrows():
    rating = ast.literal_eval(row['ratings'])
    for item in rating:
        rating_names.add(item['name'])    

rating_data = defaultdict(list) #returns a new empty list object
for index, row in tedTalks.iterrows():
    rating = ast.literal_eval(row['ratings']) # Evaluate the code before you deem it to be unsafe or not
    rating_data['Talk_ID'].append(row['Talk_ID'])
    rating_data['Speaker'].append(row['main_speaker'])
    names = set()
    for item in rating:
        rating_data[item['name']].append(item['count'])
        names.add(item['name'])
rating_data = pd.DataFrame(rating_data)
rating_data.head()


########################################################################

funnyCount = tedTalks['funny'].sum()
jawDropCount = tedTalks['jawdrop'].sum()
beautifulCount = tedTalks['beautiful'].sum()
confusingCount = tedTalks['confusing'].sum()
allReactions =pd.Series([funnyCount,jawDropCount,beautifulCount,confusingCount])

reactionsDF = pd.DataFrame({'reactions':['funny','jawdrop','beautiful','confusing'],'count': allReactions})
reactionsDF['percent']= reactionsDF['count']/reactionsDF['count'].sum() * 100
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
plt.pie(reactionsDF.percent, labels=reactionsDF.reactions,colors = colors,
        autopct='%1.1f%%', shadow=True, startangle=0)
centre_circle = plt.Circle((0,0),0.50,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.show()



#tedTalks['ratings'] = tedTalks['ratings'].apply(lambda x: ast.literal_eval(x))
#tedTalks['funny'] = tedTalks['ratings'].apply(lambda x: x[0]['count'])
#tedTalks['jawdrop'] = tedTalks['ratings'].apply(lambda x: x[-3]['count'])
#tedTalks['beautiful'] = tedTalks['ratings'].apply(lambda x: x[3]['count'])
#tedTalks['confusing'] = tedTalks['ratings'].apply(lambda x: x[2]['count'])
