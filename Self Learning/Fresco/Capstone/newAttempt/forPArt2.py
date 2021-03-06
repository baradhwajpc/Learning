# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:01:57 2018

@author: baradhwaj
"""
import nltk
import pandas as pd
import csv
import os 
import string
import re
from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.model_selection import train_test_split

from nltk.classify.util import apply_features,accuracy
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import numpy as np
del data
#Data Loading and cleanup
os.getcwd()
os.chdir("G:\\Python\\Learning\\Self Learning\\Fresco\\Capstone")
data = pd.read_csv("emails.csv",header=0,usecols=["text","spam"])
data['spam'] = pd.to_numeric(data['spam'], errors='coerce')
data = data[np.isfinite(data['spam'])]
# end of dataloading and cleanup

class SpamClassifier:
    uniqueWords = []
    classifier = 0
def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels
                
        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        tokenarray = []
        stop_words_eng = set(stopwords.words('english'))
        stop_words_fr = set(stopwords.words('french'))
        #Subject
        for t in text:
            for res in target:
                tokens = nltk.word_tokenize(t)
                tokens = list(filter(lambda x: x.lower() not in stop_words_eng and x.lower() not in stop_words_fr and x.lower()!= 'Subject' and len(x) >=3, tokens))
                tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
                dataTuple = (tokens,res)
            tokenarray.append(dataTuple)
        return tokenarray

def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels
        Return Type is a set
        """
        for tokens, labels in corpus:
            for item in tokens:
                if item not in SpamClassifier.uniqueWords:
                    SpamClassifier.uniqueWords.append(item)
        return SpamClassifier.uniqueWords
        
def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string
        
        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        features = {}
        for word in document:
            if word in SpamClassifier.uniqueWords:
                features[word] = True
            else:
                features[word] = False
        return features


def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        extractTokens = []
        extractTokens = self.extract_tokens(text, labels)
        result = self.get_features(extractTokens)
        trainedDataSet = nltk.classify.apply_features(self.extract_features,extractTokens)
        classifier = nltk.NaiveBayesClassifier.train(trainedDataSet)
        return classifier, result
    
def predict(self, text):
        """
        Returns prediction labels of given input text.
        Allowed Text can be a simple string i.e one input email, a list of emails, or a dictionary of emails identified by their labels.
        """
        #accuracyPercent = nltk.classify.accuracy(self.classifier, text)*100
        #predictedData = self.classifier.classify(text)
        #return predictedData,accuracyPercent
        
        testTokens = []
        testTokens = self.extract_tokens(text, [0,1])
        result = self.get_features(testTokens)
        testDataSet = nltk.classify.apply_features(self.extract_features,testTokens)
        accuracyPercent = nltk.classify.accuracy(self.classifier, testDataSet)*100
        predictedData = self.classifier.classify(testDataSet)
        return predictedData,accuracyPercent


if __name__ == '__main__':    
    train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                                                            data["spam"].values,
                                                            test_size = 0.25,
                                                            random_state = 50,
                                                            shuffle = True,
                                                            stratify=data["spam"].values)
    classifier = SpamClassifier()
    
    
    
    # train Method
    #Call : #classifier_model, model_word_features = classifier.train(train_X, train_Y)
    extractTokens = []
    # Call : extractTokens = self.extract_tokens(train_X, train_Y) #(text, labels)
        # Begin of Extract tokens
        tokenarray = []
        stop_words_eng = set(stopwords.words('english'))
        stop_words_fr = set(stopwords.words('french'))
            #Subject
            for t in text:
                for res in target:
                    tokens = nltk.word_tokenize(t)
                    tokens = list(filter(lambda x: x.lower() not in stop_words_eng and x.lower() not in stop_words_fr and x.lower()!= 'Subject' and len(x) >=3, tokens))
                    tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
                              dataTuple = (tokens,res)
            tokenarray.append(dataTuple)
        # End of Extract tokens
    
    
    
    result = self.get_features(extractTokens)
    trainedDataSet = nltk.classify.apply_features(self.extract_features,extractTokens)
    classifier = nltk.NaiveBayesClassifier.train(trainedDataSet)
    #End of Train Method
    
    predicted,accuracy = classifier.predict(test_X)
    print(accuracy)
    model_name = 'spam_classifier_model.pk'
    model_word_features_name = 'spam_classifier_model_word_features.pk'
    with open(model_name, 'wb') as model_fp:
        pickle.dump(classifier_model, model_fp)
    with open(model_word_features_name, 'wb') as model_fp:
            pickle.dump(model_word_features, model_fp)
    





            
stop_words_eng = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
#Subject
for t in text:
            for res in target:
                tokens = nltk.word_tokenize(t)
                tokens = list(filter(lambda x: x.lower() not in stop_words_eng and x.lower() not in stop_words_fr and x.lower()!= 'Subject' and len(x) >=3, tokens))
                tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
                dataTuple = (tokens,res)
            tokenarray.append(dataTuple)
            
            
            
            