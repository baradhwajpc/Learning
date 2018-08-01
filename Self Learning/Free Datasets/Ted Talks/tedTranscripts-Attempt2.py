# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 06:43:26 2018

@author: baradhwaj
"""

import os 
import pandas as pd
import re
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, MWETokenizer
from nltk.stem import porter, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from nltk.draw.dispersion import dispersion_plot
import string
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Ted Talks')
tedTranscripts = pd.read_csv('transcripts.csv')
# Remove pzrzznthesis in speech - All audience reactions ( applause , singing etc are under brackets ())            
# Working with first 25 scripts alone due to memory error constraints
rows = range(0,100)
# remove parethetical non-speech sounds from text
clean_parens_docs= [re.sub(r'\([^)]*\)', ' ', 
                    tedTranscripts['transcript'][row]) \
                    for row in rows]
# Get all transcprits
transcripts = range(0,25)
stopwords_en = set(stopwords.words('english'))
stopwords_punctuation = stopwords_en.union(string.punctuation) # merge set together
new_words = ["♫","'",' â€”',"(Applause)",""]
print(stopwords_punctuation)
# Split into sentence - Method 1
doc_sents = [TextBlob(clean_parens_docs[transcript])
             .sentences for transcript in transcripts]


# Split into sentence Methood 2 
doc_sents1 = [sent_tokenize(clean_parens_docs[transcript]) for transcript in transcripts]
# Word Tokenize Split sentence into words
doc_words = [TextBlob(str(doc_sents1[transcript]))
              .words for transcript in transcripts]
print('\n-----\n'.join(TextBlob(str(doc_sents1[0])).words))

# Another Method : leaves at punctuation
doc_words1 = [word_tokenize(clean_parens_docs[transcript]) \
             for transcript in transcripts]
print('\n-----\n'.join(word_tokenize(clean_parens_docs[0])))
# Another Version : wordpunct version will take care of punctuations nicely
doc_words2 = [wordpunct_tokenize(clean_parens_docs[transcript]) \
             for transcript in transcripts]
print('\n-----\n'.join(wordpunct_tokenize(clean_parens_docs[0])))

# text blob allows us to pull out interesting things in the data
talks_blob = [TextBlob(clean_parens_docs[transcript]) for transcript in transcripts]
print('\n-----\n'.join(talks_blob[1].tags))
for i in range(0,len(tedTranscripts)):
    print(talks_blob[i].sentiment)

for i in range(0,len(tedTranscripts)):
    print(talks_blob[i].noun_phrases)
#an alternative method for getting the word roots 
#This one appears to be more conservative and also more 'correct' in that it will replace 
#the ending with the correct letters instead of chopping it off. i.e. children -> child, 
#capacities -> capacity, but also, unpredictability -> unpredictability .

lemmizer = WordNetLemmatizer()
for transcript in transcripts: 
    doc = TextBlob(clean_parens_docs[transcript]).words
    for w in doc:
        print(lemmizer.lemmatize(w))
# Porter Stemmer
stemmer = nltk.stem.porter.PorterStemmer()
for transcript in transcripts: 
    doc = TextBlob(clean_parens_docs[transcript]).words
    for w in doc:
        print(stemmer.stem(w.lower()),w)
        
doc_words2 = [wordpunct_tokenize(clean_parens_docs[transcript]) \
             for transcript in transcripts]
print(type(doc_words2))
stemmer = nltk.stem.porter.PorterStemmer()
for transcript in transcripts: 
    doc = TextBlob(doc_words2[transcripts[0]]).words
    for w in doc:
        print(stemmer.stem(w.lower()),w)

from collections import Counter
from operator import itemgetter
from nltk.util import ngrams

counter = Counter()

n = 3
for doc in clean_parens_docs:
    words = TextBlob(doc).words
    bigrams = ngrams(words, n)
    counter += Counter(bigrams)

for phrase, count in counter.most_common(30):
    print('%20s %i' % (" ".join(phrase), count))