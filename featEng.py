from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier



class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

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

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line['position'] in labels:
            labels.append(line['position'])

    x_train = feat.train_feature(x['question'] for x in train)
    x_test = feat.test_feature(x['question'] for x in test)

    y_train = array(list(labels.index(x['position']) for x in train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    predictions = lr.predict(x_test)
    o = DictWriter(open("guess.csv", 'w'), ["id", "position"])
    o.writeheader()
    count = 0
    for ii, pp in zip([x['id'] for x in test], predictions):
        count+=1
        d = {'id': ii, 'position': labels[pp]}
        o.writerow(d)
    o.close()
    print count
