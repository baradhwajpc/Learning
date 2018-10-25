from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer


class SpamClassifier:
    
    def extract_tokens(self, text, target):
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
    
    def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels
        
        Return Type is a set
        """
        words = re.findall(r'\w+', corpus.lower())
        return set(words)
        
    
    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string
        
        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return {word: True for word in lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(document, errors='ignore')) if not word in stop_words}
        

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        train_size = int(len(text) * labels)
        # initialise the training and test sets
        train_set, test_set = text[:train_size], text[train_size:]
        # train the classifier
        classifier = NaiveBayesClassifier.train(train_set)
        return train_set, test_set, classifier
        
    
    def predict(self, text):
        """
        Returns prediction labels of given input text.
        Allowed Text can be a simple string i.e one input email, a list of emails, or a dictionary of emails identified by their labels.
        """
        model1 = NaiveBayesClassifier()
        return model1.predict(text)
    
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
    