# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:03:14 2018

@author: baradhwaj
"""

import os
import nltk
import pandas as pd
import re

from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from operator import itemgetter
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Ted Talks')
tedTranscripts = pd.read_csv('transcripts.csv')
rows = range(0,25)
clearedText = [re.sub(r'\([^)]*\)', ' ', tedTranscripts['transcript'][row]) for row in rows]
### Clearing text using nltk
# 1. tokenize
# 2. lemmatize
# 3. remove stop words/punctuation
# 4. vectorize
# tokenize- This is the process of splitting up the document (talk) into words.
# There are a few tokenizers in NLTK, 
# and one called wordpunct 
#Tokenizers is used to divide strings into lists of substrings. 
#For example, Sentence tokenizer can be used to find the list of sentences and 
#Word tokenizer can be used to find the list of words in strings.                      

#tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
#tokenizer.tokenize(sent_tokenize(clearedText[row]) for row in rows)
#sent_tokenize_list = [sent_tokenize(clearedText[row]) for row in rows]
# Word tokenize and remove stop words
#stop = stopwords.words('english')
#stop += ['.'," \'", 'ok','okay','yeah','ya','stuff','?']
#word_tokenize = [wordpunct_tokenize(clearedText[row]) for row in rows]

   """ 
    Takes in a corpus of documents and cleans. ONly works with multiple docs for now
    
    1. remove parentheticals
    2. tokenize into words using wordpunct
    3. lowercase and remove stop words
    4. lemmatize 
    5. lowercase and remove stop words
    OUT: cleaned text = a list (documents) of lists (cleaned word in each doc)
    """
 
def clean_text(text):    
    lemmizer = WordNetLemmatizer()
    #stemmer = porter.PorterStemmer()
    stop = stopwords.words('english')
    stop += ['.', ',',':','...','!"','?"', "'", '"',' - ',' — ',',"','."','!', ';','♫♫','♫',\
             '.\'"','[',']','—',".\'", 'ok','okay','yeah','ya','stuff', ' 000 ',' em ',\
             ' oh ','thank','thanks','la','was','wa','?','like','go',' le ',' ca ',' I '," ? ","s", " t ","ve","re"]
    cleaned_text = []    
    for post in text:
        cleaned_words = []        
        # remove parentheticals
        clean_parens = re.sub(r'\([^)]*\)', ' ', post)
        # tokenize into words
        for word  in wordpunct_tokenize(clean_parens):              
            # lowercase and throw out any words in stop words
            if word.lower() not in stop:            
                # lemmatize  to roots
                low_word = lemmizer.lemmatize(word)  
                # stem and lowercase ( an alternative to lemmatize)
                #low_word = stemmer.stem(root.lower())              
                # keep if not in stopwords (yes, again)
                if low_word.lower() not in stop:                     
                    # put into a list of words for each document
                    cleaned_words.append(low_word.lower())        
        # keep corpus of cleaned words for each document    
        cleaned_text.append(' '.join(cleaned_words))    
    return cleaned_text

# Count the maximum used phrases
cleaned_talks = clean_text(clearedText)
counter =  Counter()
n=2
for doc in cleaned_talks:
    words = TextBlob(doc).words
    bigrams = ngrams(words,n)
    counter += Counter(bigrams)

for phrase, count in counter.most_common(50):
        print('%20s %i' % (" ".join(phrase), count))

# Vectirization : Create a sparce matrix - Turn numbers into words
# This function takes each word in each document and counts the number of times the word appears. 
#You end up with each word (and n-gram) as your columns and each row is a document (talk), 
#so the data is the frequency of each word in each document.
#As  you can imagine, there will be a large number of zeros in this matrix; 
#we call this a sparse matrix.

c_vectorizer = CountVectorizer(ngram_range=(1,3), 
                                 stop_words='english', 
                                 max_df = 0.6, 
                                 max_features=10000)

    # call `fit` to build the vocabulary
c_vectorizer.fit(cleaned_talks)
    # finally, call `transform` to convert text to a bag of words
c_x = c_vectorizer.transform(cleaned_talks)


t_vectorizer = TfidfVectorizer(ngram_range=(1, 3),  
                                   stop_words='english', 
                                   token_pattern="\\b[a-z][a-z]+\\b",
                                   lowercase=True,
                                   max_df = 0.6)

# call `fit` to build the vocabulary
t_vectorizer.fit(cleaned_talks)
# finally, call `transform` to convert text to a bag of words
t_x = t_vectorizer.transform(cleaned_talks)

def topic_mod_lda(data, topics,iters,ngram_min,ngram_max,max_df,max_feats):
    
    vectorizer = CountVectorizer(ngram_range=(ngram_min,ngram_max), 
                                 stop_words='english', 
                                 max_df = max_df, 
                                 max_features=max_feats)
    vectorizer._validate_vocabulary()

        # call `fit` to build the vocabulary
    vect_data = c_vectorizer.fit_transform(data)
    # LDA will provide a strength score for each topic
    lda = LatentDirichletAllocation(n_components=topics,
                                    max_iter=iters,
                                    random_state=42,
                                    learning_method='online',
                                    n_jobs=-1)
    lda_data = lda.fit_transform(vect_data)
    
    for ix, topic in enumerate(lda.components_):
            print("Topic ", ix)
            print(" ".join([vectorizer.get_feature_names()[i]
                        for i in topic.argsort()[:-20 - 1:-1]]))
            
    return vectorizer, vect_data, lda, lda_data
  
  vect_mod, vect_data, lda_mod, lda_data = topic_mod_lda(cleaned_talks, topics=20,
                                             iters=100, ngram_min=1, ngram_max=2, 
                                             max_df=0.5, max_feats=2000)