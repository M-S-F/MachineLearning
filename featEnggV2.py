from csv import DictReader, DictWriter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
import yaml
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
from collections import Counter
from math import sqrt
import ast
from nltk.corpus import wordnet as wn
from itertools import product
import nltk
import cPickle as pickle
import string, re
from nltk.corpus import stopwords
from pattern.en import tag
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import LancasterStemmer

kTOKENIZER = WordPunctTokenizer()
stop = stopwords.words('english')
regex = re.compile('[%s]' % re.escape(string.punctuation))
stemmer = LancasterStemmer()

def tokenize(text):
    d = defaultdict(int)
    tokens = kTOKENIZER.tokenize(text)
    filtered_words = [w for w in tokens if not w in stop]###Removing Stop Words
    filtered_words = [regex.sub('', w) for w in filtered_words]
    for word in filtered_words:
        word = stemmer.stem(str(word).lower())
        d[word]+=1
    d[ngrams(tokens,2)]+=1
    return d

def ngrams(lst, n):
    tlst = lst
    while True:
        a, b = tee(tlst)
        l = tuple(islice(a, n))
        if len(l) == n:
            yield l
            next(b)
            tlst = b
        else:
            break

class classify:
    
    
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y = []

        self.test = []
        self.train = []        

        self.quesFeat = {}        
        self.predictions = []
        
        self.cv_train = []
        self.cv_test = []
        self.cv_y_train = []
        self.cv_y_test = []
        self.v = DictVectorizer(sparse=False)

    def questionFeatures(self):
        questions = list(DictReader(open("questions.csv", 'rU')))
        self.train = list(DictReader(open("train.csv", 'rU')))
        positions = {}
        for each in self.train:
            positions.update({each['question']: float(each['position'])})

        countries = []
        cities = []
        states = []
        with open("countries.txt", "r") as fp:
            countries.append(stemmer.stem(str(regex.sub('', fp.readline())).lower()))
        fp.close()
        with open("uscities.txt", "r") as fp:
            cities.append(stemmer.stem(str(regex.sub('', fp.readline())).lower()))
        fp.close()
        with open("usstates.txt", "r") as fp:
            states.append(stemmer.stem(str(regex.sub('', fp.readline())).lower()))
        fp.close()
        
        for ques in questions:
            unigrams = {}
            unigrams['_length_'] = len(ques['questionText'])
            unigrams['_cat_'] = ques['cat']
            unigrams['_answer_'] = ques['answer']
            unigrams['_answerLength_'] = len(ques['answer'])
            
            if ques['question'] in positions.keys():
                unigrams['_position_'] = positions[ques['question']]
            
            bestWord = ""
            bestPos = 0
        
            ## Just the nouns for words which are 5 positions from our position
            for pos, word in ast.literal_eval(ques['unigrams']).items():
                
                if ques['question'] in positions.keys():
                    
                    #Best word closest to the question position
                    if pos<=positions[ques['question']]:
                        bestWord = word
                        bestPos = pos

                for w,t in tag(str(word)):
                    if str(t) in ['NN', 'NNS', 'NNP', 'NNPS']:
                        if "_nouns_" in unigrams.keys():
                            unigrams["_nouns_"]+=1
                        else:
                            unigrams["_nouns_"]=1
                    elif str(t) in ['VB','VBD','VBG','VBN','VBP','VBZ']:
                        if "_verbs_" in unigrams.keys():
                            unigrams["_verbs_"]+=1
                        else:
                            unigrams["_verbs_"]=1
    
                try:
                    if isinstance(int(str(word)), int):
                        if len(str(word)) <= 3:
                            if "_number_" in unigrams.keys():
                                unigrams["_number_"]+=1
                            else:
                                unigrams["_number_"]= 1
                        else:
                            if "_year_" in unigrams.keys():
                                unigrams["_year_"]+=1
                            else:
                                unigrams["_year_"]=1
                            
                except:
                    None
            
            """
            ### Lower case and stemming unigrams
            
            unigrams = dict(unigrams.items()+tokenized_words.items())
            
            with open("unigrams.txt",'a') as fp:
                pickle.dump(unigrams, fp)
            fp.close()
            """
            #### Stemming countries with removed punctuation
            #tokenized_words = tokenize(ques['questionText'])
            for element in countries:
                if element in unigrams.keys():
                    unigrams[element]+=1
                else:
                    unigrams[element]=1
            for element in cities:
                if element in unigrams.keys():
                    unigrams[element]+=1
                else:
                    unigrams[element]=1
            for element in states:
                if element in unigrams.keys():
                    unigrams[element]+=1
                else:
                    unigrams[element]=1

            #unigrams = dict(unigrams.items()+tokenized_words.items())
            unigrams['_bestWord_'] = bestWord
            unigrams['_bestWordPos_'] = bestPos
            self.quesFeat[ques['question']] = unigrams

    def readData(self, accuracy):
        print "In readData"
        self.train = list(DictReader(open("train.csv", 'rU')))
        self.test = list(DictReader(open("test.csv", 'rU')))
        ind = 0
        
        # training set
        for each in self.train:
            features = {}
            features = self.quesFeat[each['question']]
            features['_user_'] = each['user']
            features['_question_'] = each['question']
            
            if(accuracy==False):
                self.x_train.append(features)
                self.y.append(float(each['position']))
            else:
                if(ind%5 == 0):
                    self.cv_test.append(features)
                    self.cv_y_test.append(float(each['position']))
                else:
                    self.cv_y_train.append(float(each['position']))
                    self.cv_train.append(features)
                ind+=1
            
        ## test set
        if(accuracy == False):
            for each in self.test:
                features = {}
                features = self.quesFeat[each['question']]
                features['_user_'] = each['user']
                features['_question_'] = each['question']
                self.x_test.append(features)
            self.x_train = self.v.fit_transform(self.x_train)
            self.x_test = self.v.transform(self.x_test)
        else:
            self.cv_train = self.v.fit_transform(self.cv_train)
            self.cv_test = self.v.transform(self.cv_test)
        
                
    def predict(self):
        print "In Predict"
        clf = linear_model.Lasso(alpha=0.01)
        clf = clf.fit(self.x_train, self.y)
        self.predictions = clf.predict(self.x_test)

    def writePredictions(self):
        print "In writePredictions"
        o = DictWriter(open("predictions.csv", 'w'), ["id", "position"])
        o.writeheader()
        for ii, pp in zip([x['id'] for x in self.test], self.predictions):
            d = {'id': ii, 'position': pp}
            o.writerow(d)
    
    def getAccuracy(self):
        
        print "In getAccuracy"
        
        clf = linear_model.Lasso(alpha=0.01)
        clf = clf.fit(self.cv_train, self.cv_y_train)
        self.predictions = clf.predict(self.cv_test)
        
        acc = 0
        ind = 0
        for pred in self.predictions:
            if(pred == self.cv_y_test[ind]):
                acc+=1
            ind+=1
        print "Accuracy correct/total : "
        print float(acc)/float(ind)
        
        print "Mean Square Error cv set is : "
        print mean_squared_error(self.cv_y_test, self.predictions)
        
        
if __name__ == "__main__":
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--accuracy', type=bool, default=False, help='test accuracy')
    args = parser.parse_args()
    
    obj = classify()
    obj.questionFeatures()
    obj.readData(args.accuracy)
    
    if(args.accuracy == True):
        obj.getAccuracy()
    else:
        obj.predict()
        obj.writePredictions()
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    
    # play beep after finishing
    #import os
    #os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 1, 150))
