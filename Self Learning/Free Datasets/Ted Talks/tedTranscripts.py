# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 07:41:50 2018

@author: baradhwaj
"""
import os
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cross_validation import train_test_split
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

from nltk.corpus import stopwords # Import stopwords
from nltk import sent_tokenize

os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Ted Talks')
tedTranscripts = pd.read_csv('transcripts.csv')

feature_algo_wo_stopwords = TfidfVectorizer(stop_words='english',ngram_range=(1,1),use_idf=True,min_df = 0.05,max_df = 0.3)
text_features_wo_stopwords = feature_algo_wo_stopwords.fit_transform(tedTranscripts['transcript'])
text_features__wo_stopwords_raw_matrix = text_features_wo_stopwords.toarray()
feature_algo_wo_stopwords.vocabulary_


freq = nltk.FreqDist(feature_algo_wo_stopwords.vocabulary_)
#freq.plot(100, cumulative=False)

## The word cloud decipts the most words used in all speeches
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='lightgrey',
                      max_words=100,
                      colormap='Greens').generate_from_frequencies(freq)
fig = plt.figure(1,figsize=(12,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()

### Introduct 
# Lets split the data to train and test

# Split 
#tedTranscripts_70 = tedTranscripts.sample(frac=0.7)
crieteria = np.random.rand(len(tedTranscripts)) < 0.7
train_tedTalk = tedTranscripts[crieteria]
test_tedTalk = tedTranscripts[~crieteria]

sentAnalyser = SentimentIntensityAnalyzer()
for sentence in train_tedTalk['transcript']:
    scores = sentAnalyser.polarity_scores(sentence)
    for score in scores:
        print('{0},{1}, '.format(score,scores[score],end=''))
    print()
###############################################################################
 
 
#####################3 NAive Bayess classifier ################################    
stopwords_set = set(stopwords.words("english"))
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
for trainTalk in train_tedTalk['transcript']:
    tokens = word_tokenize(trainTalk)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    more_stopwords = "applause"
    stop_words.update(more_stopwords.split())
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    print(stemmed)
#classifier = NaiveBayesClassifier.train(tedTranscripts_70['transcript'])
#print 'accuracy:', nltk.classify.util.accuracy(classifier, tedTranscripts_70['transcript'])
#classifier.show_most_informative_features()

