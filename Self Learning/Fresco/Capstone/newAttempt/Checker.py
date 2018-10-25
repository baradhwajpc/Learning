from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer
from sklearn.model_selection import train_test_split


class SpamClassifier:    

    def extract_tokens(self, text, target):
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
                tokens = list(filter(lambda x: x.lower() not in stop_words, tokens)) # remove stop words
                tokens = [w.lower() for w in tokens if len(w) >=3]
                tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
                tupleobj = (tokens,l)     
            tokenarray.append(tupleobj)
            print(i)
            i=i+1
        return tokenarray
    
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
        #https://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis
        features = {}  
        for word in nltk.word_tokenize(document):
            features[word] = True
        return features


        

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        tuple_extract_tokens = []
        tuple_extract_tokens = self.extract_tokens(text, labels)
        print(tuple_extract_tokens)
        #result_set = self.get_features(tuple_extract_tokens)

        #for text, label in tuple_extract_tokens:
            #print(self.extract_features(text))
        #training_set = nltk.classify.apply_features(self.extract_features,text)
        #print(training_set)
        #featuresets = [self.extract_features(text) for text in tuple_extract_tokens]
        #size = int(len(featuresets) * 0.1)
        #train_set, test_set = featuresets[size:], featuresets[:size]
  
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return classifier, result_set
    
    def predict(self, text):
        """
        Returns prediction labels of given input text.
        Allowed Text can be a simple string i.e one input email, a list of emails, or a dictionary of emails identified by their labels.
        """
    
if __name__ == '__main__':
    
    data = pd.read_csv('emails.csv')
    train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values,
                                                            data["spam"].values,
                                                            test_size = 0.98,
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
