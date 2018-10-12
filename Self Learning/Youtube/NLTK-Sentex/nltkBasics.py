# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 09:14:56 2018

@author: baradhwaj
"""

from nltk.tokenize import sent_tokenize,word_tokenize
text = 'Hey hi Mr.Baradh, how are you. What are you doing now?'
print(sent_tokenize(text))
print(word_tokenize(text))
for i in word_tokenize(text):
    print(i)

# Stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
    
    example_sentence = "This is an example showing off stop words filtration."
    stop_words = set(stopwords.words('english')) # var containing the set of stop words in english language
    filtered_words = []
    words = word_tokenize(example_sentence)
    
    for w in words:
        if w not in stop_words:
            filtered_words.append(w)
            
    print(filtered_words)
    
filtered_words_exp = [w for w in words if not w in stop_words]
print(filtered_words_exp)

# Stemming
#Taking the root word from words - writing --> write . Remove ing
# Removing Fixations in words so tht we can cleanup redundant words in the table / db
# Porter Stemmer 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps  = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
for w in example_words:
    print(ps.stem(w))
new_text ="It is important to be pythonly while you are pythoning with python.All pythonershave pythoned poorly atleast once."
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))

# Part of speech 
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#Train 
train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text) # Train the tokenizer with a speech
tokenized = custom_sent_tokenizer.tokenize(sample_text)# Tokenize  the sample_text with
def proces_text ():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tagged(words)
            print(tagged)# prints word & part of speech
    except Exception as e:
        print(str(e))
        
# Chunking
    

import pandas as pd

df_e = pd.DataFrame()




import pandas
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()

dfEnc = pandas.DataFrame({
    'pets': ['cat', 'dog', 'cat', 'monkey', 'dog', 'dog'], 
    'owner': ['Champ', 'Ron', 'Brick', 'Champ', 'Veronica', 'Ron'], 
    'location': ['San_Diego', 'New_York', 'New_York', 'San_Diego', 'San_Diego', 
                 'New_York']
})
dfEnc.apply(le.fit_transform)
dfE1 = pandas.DataFrame({
    'c1': ['a', 'p', 'b', 'q', 'c', 'a']})
df_encoded = dfE1.apply(le.fit_transform)

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
data = ['a','b','c','p','q']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)



from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['a','b','c','p','q'])
le.classes_
le.transform(['a','q','c','p']) 
#array([0, 0, 1, 2]...)
le.inverse_transform([0, 0, 1, 2])
#array([1, 1, 2, 6])

