from csv import DictReader, DictWriter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import json
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

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
        questions = list(DictReader(open("questions.csv", 'r')))
        for ques in questions:
            unigrams = {}
            #unigrams = json.loads(ques['unigrams'])
            unigrams['_length_'] = len(ques['questionText'])
            unigrams['_cat_'] = ques['cat']
            unigrams['_answer_'] = ques['answer']    
            self.quesFeat[ques['question']] = unigrams     
            
    def readData(self, accuracy):
        print "In readData"
        self.train = list(DictReader(open("train.csv", 'r')))
        self.test = list(DictReader(open("test.csv", 'r')))
        v = DictVectorizer(sparse=False)        
        ind = 0
        
        # training set
        for each in self.train:
            features = {}
            features = self.quesFeat[each['question']]
            features['_user_'] = each['user']
            features['_question_'] = each['question']
            #features['_answer_'] = each['answer'] 
            
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
        #clf = tree.DecisionTreeClassifier()
        clf = DecisionTreeRegressor(max_depth=100)
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
        #clf = tree.DecisionTreeClassifier()
        
        clf = DecisionTreeRegressor(max_depth=5)
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
    import os
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( 1, 150))