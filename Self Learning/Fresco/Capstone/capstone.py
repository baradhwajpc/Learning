# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:32:47 2018

@author: baradhwaj
"""
import nltk
import pandas as pd
import csv
import os 
import string
import re

#Data Loading
os.getcwd()
os.chdir("G:\\Python\\Learning\\Self Learning\\Fresco\\Capstone")
rawdata = pd.read_csv("emails.csv",header=0)

messages = rawdata['text']
stop_words = set(stopwords.words('english'))
morewords = ['Subject:']
stop_words.update(morewords)

new_words = []
        
    def extract_tokens(text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels
                
        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        tokenarray = []
        print(len(text))
        i=0
        stop_words = set(stopwords.words('english'))
        for t in text:
            for l in target:
                tokens = nltk.word_tokenize(t) # tokenize the text'
                tokens = list(filter(lambda x: x.lower() not in stop_words and len(x) >=3 and x.lower() != "subject", tokens)) # remove stop words
                #tokens = [w.lower() for w in tokens if len(w) >=3]
                
                tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
                dataTuple = (tokens,l)     
            tokenarray.append(dataTuple)
            i=i+1
        return tokenarray
extract_tokens(messages,[0,1])    
    
    def get_features(self, corpus):
        """ 
        returns a Set of unique words in complete corpus. 
        parameters:- corpus: tokenized corpus along with target labels
        
        Return Type is a set
        """
        unique_word_list = []
        for tokens, labels in corpus:
            for item in tokens:
                if item not in unique_word_list:
                    unique_word_list.append(item)
        return unique_word_list
    
    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string
        
        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        
    
    def predict(self, text):
        """
        Returns prediction labels of given input text. 
        Allowed Text can be a simple string i.e one input email, a list of emails, or a dictionary of emails identified by their labels.
        """
        
    
    
if __name__ == '__main__':
    
    data = pd.read_csv('emails.csv')
    train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                                                            data["spam"].values,
                                                            test_size = 0.25,
                                                            random_state = 50,
                                                            shuffle = True,
                                                            stratify=data["spam"].values)
    classifier = SpamClassifier()
    classifier_model, model_word_features = classifier.train(train_X, train_Y)
    model_name = 'spam_classifier_model.pk'
    model_word_features_name = 'spam_classifier_model_word_features.pk'
    with open(model_name, 'wb') as model_fp:
        pickle.dump(classifier_model, model_fp)
    with open(model_word_features_name, 'wb') as model_fp:
            pickle.dump(model_word_features, model_fp)

            
            


    def extract_tokens(text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels
                
        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        stop_words = set(stopwords.words('english'))
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]') #remove punctuation
        text = regex.sub(' ', text)
        tokens = nltk.word_tokenize(text) # tokenize the text
        tokens = list(filter(lambda x: x.lower() not in stop_words, tokens)) # remove stop words
        tokens = [w.lower() for w in tokens if len(w) >=3]
        tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        return tokens
extract_tokens(messages,[0,1])

tupArray = []
for word in messages:
    for l in [0,1]:
        new_word = re.sub(r"[^a-zA-Z0-9]+",' ', l)
        if new_word != '' and new_word not in stop_words and len(new_word) >=3:
            tokens = nltk.word_tokenize(new_words.lower())
            tokenizedTuple = (token,l)
        tupArray.append(tokenizedTuple)   

