from csv import DictReader, DictWriter
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
import json

class classify:
    
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.test = []
        self.train = []
        self.y = []
        self.quesFeat = {}        
        self.predictions = []
    
    def questionFeatures(self):
        questions = list(DictReader(open("questions.csv", 'r')))
        for ques in questions:
            unigrams = {}
            #unigrams = json.loads(ques['unigrams'])
            unigrams['_length_'] = len(ques['questionText'])
            unigrams['_cat_'] = ques['cat']
            unigrams['_answer_'] = ques['answer']    
            self.quesFeat[ques['question']] = unigrams     
            
    def readData(self):
        self.train = list(DictReader(open("train.csv", 'r')))
        self.test = list(DictReader(open("test.csv", 'r')))
        
        for each in self.train:
            self.y.append(each['position'])
            features = {}
            features = self.quesFeat[each['question']]
            features['_user_'] = each['user']
            #features['_answer_'] = each['answer'] 
            self.x_train.append(features)

        for each in self.test:
            features = {}
            features = self.quesFeat[each['question']]
            features['_user_'] = each['user']        
            self.x_test.append(features)

        v = DictVectorizer(sparse=False)
        self.x_train = v.fit_transform(self.x_train)
        self.x_test = v.transform(self.x_test)
        
    def predict(self):        
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.x_train, self.y)
        self.predictions = clf.predict(self.x_test)

    def writePredictions(self):
        o = DictWriter(open("predictions.csv", 'w'), ["id", "position"])
        o.writeheader()
        for ii, pp in zip([x['id'] for x in self.test], self.predictions):
            d = {'id': ii, 'position': pp}
            o.writerow(d)

if __name__ == "__main__":
    obj = classify()
    obj.questionFeatures()
    obj.readData()
    obj.predict()
    obj.writePredictions()
