from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

from flask import current_app
import pickle, re
import collections
from nltk.corpus import stopwords


class SpamClassifier:
    uniqueWords =[]
    #uniqueWords = set([]) 

    def load_model(self, model_name):
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],model_name +'_word_features.pk')
        with open(model_file, 'rb') as mfp:
            self.classifier = pickle.load(mfp)
        with open(model_word_features_file, 'rb') as mwfp:
            self.word_features = pickle.load(mwfp)


    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels

        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        stop_words_eng = set(stopwords.words('english'))
        stop_words_fr = set(stopwords.words('french'))
        dataTuple = ()
        tokenArray = []
        
        #Subject
        for t in text:
                for res in target:
                    tokens = set(nltk.word_tokenize(t))
                    tokens = list(filter(lambda x: x.lower() not in stop_words_eng and x.lower() 
                            not in stop_words_fr and x.lower()!= 'subject' and len(x) >=3, tokens)) # and x.lower()!= 'subject' 
                    tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
                    dataTuple = (tokens,res)
                tokenArray.append(dataTuple)
        return tokenArray

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
        return set(SpamClassifier.uniqueWords)
        

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
                i+=1
            else:
                features[word] = False
                j+=1
        return features

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        tupArray = self.extract_tokens(train_X, train_Y)
        result = self.get_features(tupArray)
        trainDataSet = nltk.classify.apply_features(self.extract_features,tupArray)
        classifier = nltk.NaiveBayesClassifier.train(trainDataSet)
        print(nltk.classify.accuracy(classifier, trainDataSet))
        print(classifier.show_most_informative_features(10))
        return classifier,result

    def predict(self, text):
        """
        Returns prediction labels of given input text.
        """
         return self.classifier.classify(text)
         
         
         
#del uniqueWords

if __name__ == '__main__':
    
    os.getcwd()
    os.chdir("G:\\Python\\Learning\\Self Learning\\Fresco\\Capstone")
    data = pd.read_csv("emails.csv",header=0,usecols=["text","spam"])
    type(data['text'][0])
    type(data['spam'][0])

    data['spam'] = pd.to_numeric(data['spam'], errors='coerce')
    data = data.dropna()
    type(data['text'][0])
    type(data['spam'][0])
    data['spam'] = data['spam'].astype(int)
    newData = data[:500]
    train_X, test_X, train_Y, test_Y = train_test_split(newData["text"],
                                                            newData["spam"],
                                                            test_size = 0.97,
                                                            random_state = 30,
                                                            shuffle = True,
                                                            stratify=newData["spam"])
    
    classifier = SpamClassifier()
    classifier_model, model_word_features = classifier.train(train_X, train_Y)
    results = classifier_model.classify_many(testDataSet)

    classifier_model.show_most_informative_features(10)

    
    
    
    
    pd.Series(['text', 'spam']).isin(data.columns).all()
    twoColsExist = pd.Series(['text', 'spam']).isin(data.columns).all()
    message='ppp'
    if(not twoColsExist):
        message = 'Only 2 columns allowed: Your input csv file has '+len(data.columns)+ 'number of columns.'
    if data.spam.dtype == 'int32':
        print(1)

    
    
   my_list = [str(x) for x in data['spam'].unique() if x != '0' and x != '1']
   str1 = ','.join(my_list)
   len(data)
   len(data[data['text'].str.startswith("Subject:")])

    # Predict
    
    train_XX, test_XX, train_YY, test_YY = train_test_split(data["text"].values,
                                                            data["spam"].values,
                                                            test_size = 0.02,
                                                            random_state = 25,
                                                            shuffle = True,
                                                            stratify=data["spam"].values)
    testTokens = []
    testTokens = classifier.extract_tokens(test_XX, test_YY)
    testresutlt = classifier.get_features(testTokens)
    testDataSet = nltk.classify.apply_features(classifier.extract_features,testTokens)
    
    classifier_model.classify(gender_features('Neo'))
    classifier_model.classify(test_XX)

    accuracyPercent = nltk.classify.accuracy(classifier_model, testTokens)*100
    print('Done')
    
    
    
    
    
    
    
    stop_words_eng = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    dataTuple = ()
    tokenArray = []
    text = train_X
    target = train_Y
        #Subject
    for t in text:
        for res in target:
            tokens = set(nltk.word_tokenize(t))
            tokens = list(filter(lambda x: x.lower() not in stop_words_eng and x.lower() 
                     not in stop_words_fr and x.lower()!= 'subject' and len(x) >=3, tokens)) # and x.lower()!= 'subject' 
            tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
            dataTuple = (tokens,res)
        tokenArray.append(dataTuple)
    
    
    
    
    stop_words_eng = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    dataTuple = ()
    tokenArray = []
    text = train_X
    target = train_Y
        #Subject
    for t in text:
        for res in target:
            tokens = set(nltk.word_tokenize(t))
            tokens = list(filter(lambda x: x.lower() not in stop_words_eng and x.lower() 
                     not in stop_words_fr and x.lower()!= 'subject' and len(x) >=3, tokens)) # and x.lower()!= 'subject' 
            tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
            dataTuple = (tokens,res)
        tokenArray.append(dataTuple)