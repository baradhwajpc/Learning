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
freq.plot(30, cumulative=False)

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

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
for trainTalk in train_tedTalk['transcript']: 
    #text = tokenizer.tokenize(trainTalk)
    sentences = sent_tokenize(trainTalk)
    # Get words from sentence
    tokens = word_tokenize(sentences)
    # Remove non alphabets in words
    words = [word for word in tokens if word.isalpha()]
    print(words)


#classifier = NaiveBayesClassifier.train(tedTranscripts_70['transcript'])
#print 'accuracy:', nltk.classify.util.accuracy(classifier, tedTranscripts_70['transcript'])
#classifier.show_most_informative_features()