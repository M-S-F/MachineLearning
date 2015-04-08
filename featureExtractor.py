from csv import DictReader, DictWriter
from collections import defaultdict
from collections import Counter

import numpy as np
import argparse
import nltk
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeRegressor
from nltk.tokenize import WordPunctTokenizer

kTOKENIZER = WordPunctTokenizer()

def tokenize(text):
    d = defaultdict(int)
    tokens = kTOKENIZER.tokenize(text)
    for word in tokens:
        d[word]+=1
    #d[ngrams(tokens,2)]+=1
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

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer(tokenizer = tokenize, stop_words = "english")

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    train_data = list(DictReader(open("train.csv", 'r')))
    question_data = list(DictReader(open("question.csv" , 'r')))
    test = list(DictReader(open("test.csv", 'r')))
    
    feat = Featurizer()
    
    feature_data = {}
    for line in train_data:
        data = {}
        question = line['question']
        if question is not None:
            data.update({'user': line['user']})
            for questionrow in question_data:
                if line['question'] == questionrow['question']:
                    data.update({'_cat_': questionrow['cat']})
                    data.update({'unigrams': questionrow['unigrams']})
                    break

            feature_data.update({question:data})



    x_train = feat.train_feature(x['question'] for x in train_data).toarray()
    x_test = feat.test_feature(x['id'] for x in test).toarray()
    
    y_train = array(list(labels.index(x['position']) for x in train))
    
    # Train classifier
    dtr = DecisionTreeRegressor(max_depth=3)
    dtr.fit(x_train, y_train)
    
    #feat.show_top10(dtr, labels)
    
    predictions = dtr.predict(x_test)
    print predictions
    o = DictWriter(open("predictions.csv", 'w'), ["id", "position"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'position': labels[pp]}
        o.writerow(d)

