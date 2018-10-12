# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 19:58:48 2018

@author: baradhwaj
"""

import pandas as pd
import csv
import os 
#Data Loading
os.getcwd()
os.chdir("G:\\Python\\Learning\\Self Learning\\Fresco\\NLP Learning\\Unstructured")
#rawdata = pd.read_csv("train.csv")

messages = [line.rstrip() for line in open('dataset.csv')]
print (len(messages))
#Appending column headers
messages = pd.read_csv('dataset.csv', sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"])
data_size=messages.shape
print(data_size)
messages_col_names=list(messages.columns)
print(messages_col_names)
print(messages.groupby('label').describe())
print(messages.head(3))

message_target=messages['label'] 
print(message_target)

import nltk
from nltk.tokenize import word_tokenize
def split_tokens(message):
  message=message.lower()
  message = unicode(message, 'utf8') #convert bytes into proper unicode
  word_tokens =word_tokenize(message)
  return word_tokens
messages['tokenized_message'] = messages.apply(lambda row: split_tokens(row['message']),axis=1)

#lemmesation
from nltk.stem.wordnet import WordNetLemmatizer
def split_into_lemmas(message):
    lemma = []
    lemmatizer = WordNetLemmatizer()
    for word in message:
        a=lemmatizer.lemmatize(word)
        lemma.append(a)
    return lemma

messages['lemmatized_message'] = messages.apply(lambda row: split_into_lemmas(row['tokenized_message']),axis=1)
print('Tokenized message:',messages['tokenized_message'][11])
print('Lemmatized message:',messages['lemmatized_message'][11])

#stop words

from nltk.corpus import stopwords
def stopword_removal(message):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    filtered_sentence = ' '.join([word for word in message if word not in stop_words])
    return filtered_sentence

messages['preprocessed_message'] = messages.apply(lambda row: stopword_removal(row['lemmatized_message']),axis=1)
Training_data=pd.Series(list(messages['preprocessed_message']))
Training_label=pd.Series(list(messages['label']))

#TDM
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tf_vectorizer = CountVectorizer(ngram_range=(1, 2),min_df = (1/len(Training_label)), max_df = 0.7)
Total_Dictionary_TDM = tf_vectorizer.fit(Training_data)
message_data_TDM = Total_Dictionary_TDM.transform(Training_data)

#Term Frequency Inverse Document Frequency (TFIDF)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),min_df = (1/len(Training_label)), max_df = 0.7)
Total_Dictionary_TFIDF = tfidf_vectorizer.fit(Training_data)
message_data_TFIDF = Total_Dictionary_TFIDF.transform(Training_data)



