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
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Ted Talks')
tedTalks = pd.read_csv('tedTalks.csv')

tedTalks['film_date']=pd.to_datetime(tedTalks['film_date'],unit='s')
tedTalks['published_date']=pd.to_datetime(tedTalks['published_date'],unit='s')

tedTalks['ratings'] = tedTalks['ratings'].apply(lambda x: ast.literal_eval(x))

tedTalks['funny'] = tedTalks['ratings'].apply(lambda x: x[0]['count'])
tedTalks['jawdrop'] = tedTalks['ratings'].apply(lambda x: x[-3]['count'])
tedTalks['beautiful'] = tedTalks['ratings'].apply(lambda x: x[3]['count'])
tedTalks['confusing'] = tedTalks['ratings'].apply(lambda x: x[2]['count'])
tedTalks.head()

# Ted talks filimed before ted was established
print(tedTalks.loc[(tedTalks['film_date'].dt.year < 1984),['title','main_speaker','film_date']])

tedTalks['abbr']= tedTalks['main_speaker'].apply(lambda x:x[:3])
f, ax = plt.subplots(figsize=(20, 18))
sns.factorplot(x="abbr", y="views", data=tedTalks.nlargest(15,'views'), kind="bar",size=7)

#
tedTalks['film_year'] = tedTalks['film_date'].dt.year
del  tedTalks['eventCount']
tedTalksEventCount = tedTalks[['title', 'event']].groupby('event').count().reset_index()


tedTalks['year'] = tedTalks['film_date'].dt.year
year_df = pd.DataFrame(tedTalks['year'].value_counts().reset_index())
year_df.columns = ['year', 'talks']
f, ax = plt.subplots(figsize=(20, 18))
sns.pointplot(x='year', y='talks', data=year_df)


plt.figure(figsize=(15,5))
sns.barplot(x='occupation', y='appearances', data=tedTalks.head(10))
plt.show()

## Lengthy speakers and thier profession

# Views vs Years
plt.figure(figsize=(20,10))
sns.swarmplot(x="year", y="views", data=tedTalks)

# Year vs Comments
plt.figure(figsize=(20,10))
sns.swarmplot(x="year", y="comments", data=tedTalks)
# Year vs Duration
plt.figure(figsize=(20,10))
sns.swarmplot(x="year", y="duration", data=tedTalks)

# Year vs Language
plt.figure(figsize=(20,10))
sns.violinplot(x="year", y="languages", data=tedTalks)


sns.pointplot(x="year", y="survived", hue="class", data=tedTalks);

# Tag Cloud for professions
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

### New plots in matplot lib


plt.figure(figsize=(16,8))
# plot chart
ax1 = plt.subplot(121, aspect='equal')
reactions=['funny','jawdrop','beautiful','confusing']
tedTalks['totalReactions']  = np.sum(tedTalks['funny']+tedTalks['jawdrop']+tedTalks['beautiful']+tedTalks['confusing']
tedTalks.plot(kind='pie', y = 'totalReactions', ax=ax1, #autopct='%1.1f%%', 
startangle=90, shadow=False, 
#labels=reactions, 
legend = False, fontsize=14)



import plotly.plotly as py
import plotly.graph_objs as go
trace = go.Pie(values=tedTalks.speaker_occupation)


fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()