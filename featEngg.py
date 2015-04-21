from csv import DictReader, DictWriter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
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
from pattern.en import tag
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

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

    def questionFeatures(self):
        questions = list(DictReader(open("questions.csv", 'rU')))
        self.train = list(DictReader(open("train.csv", 'rU')))
        positions = {}
        for each in self.train:
            positions.update({each['question']: float(each['position'])})

        for ques in questions:
            unigrams = {}
            unigrams['_length_'] = len(ques['questionText'])
            unigrams['_cat_'] = ques['cat']
            unigrams['_answer_'] = ques['answer']
            unigrams['_answerLength_'] = len(ques['answer'])
            bestWord = ""
            bestPos = 0
            for pos, word in ast.literal_eval(ques['unigrams']).items():
                if ques['question'] in positions.keys():
                    
                    #Best word closest to the question position
                    if pos<=positions[ques['question']]:
                        bestWord = word
                        bestPos = pos
                    
                    ###Number and Year feature
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
                
                    ## Just the nouns
                    for w,t in tag(str(word)):
                        if str(t) in ['NN', 'NNS', 'NNP']:
                            if "_nouns_" in unigrams.keys():
                                unigrams["_nouns_"]+=1
                            else:
                                unigrams["_nouns_"]=1
        
                    ### Lower case and stemming unigrams
                    stemmed_word = stemmer.stem(str(word).lower())
                    if stemmed_word in unigrams:
                        unigrams[stemmed_word]+=1
                    else:
                        unigrams[stemmed_word]=1
                            
                    #### Stemming countries with removed punctuation

        
            unigrams['_bestWord_'] = bestWord
            unigrams['_bestWordPos_'] = bestPos
            self.quesFeat[ques['question']] = unigrams

    def readData(self, accuracy):
        print "In readData"
        self.train = list(DictReader(open("train.csv", 'rU')))
        self.test = list(DictReader(open("test.csv", 'rU')))
        v = DictVectorizer(sparse=False)        
        ind = 0
        
        # training set
        for each in self.train:
            features = {}
            features = self.quesFeat[each['question']]
            #features['_user_'] = each['user']
            #features['_question_'] = each['question']
            
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
            self.x_train = v.fit_transform(self.x_train)
            self.x_test = v.transform(self.x_test)
        else:
            self.cv_train = v.fit_transform(self.cv_train)
            self.cv_test = v.transform(self.cv_test)
        
                
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
    parser.add_argument('--accuracy', type=bool, default=False,
                        help='test accuracy')
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
