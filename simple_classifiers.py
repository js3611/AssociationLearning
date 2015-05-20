__author__ = 'joschlemper'

import utils
import kanade_loader as loader
import numpy as np
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

class SimpleClassifier(object):
    def __init__(self, classifier='logistic', train_x=None, train_y=None):
        if classifier == 'knn':
            clf = KNeighborsClassifier(7, weights='uniform')
        elif classifier == 'logistic':
            clf = LogisticRegression()
        else:
            clf = svm.LinearSVC()

        if not train_x == None:
            train_x, train_y = self.__get_values(train_x, train_y)
            clf.fit(train_x, train_y)

        self.clf = clf

    def retrain(self, train_x, train_y):
        train_x, train_y = self.__get_values(train_x, train_y)
        self.clf.fit(train_x, train_y)

    def get_score(self, test_x, test_y):
        test_x, test_y = self.__get_values(test_x, test_y)
        guess_y = self.clf.predict(test_x)
        print metrics.classification_report(test_y, guess_y)
        print metrics.confusion_matrix(test_y, guess_y)
        return np.sum(guess_y == test_y) * 1. / len(guess_y)

    def __get_values(self, x, y):
        if utils.isSharedType(x):
            x = x.get_value(borrow=False)
            y = y.eval()
        return x, y


def get_score(train_x, train_y, test_x, test_y, classifier='logistic'):
    '''
    param -> float [0, 1]

    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param classifier:
    :return:
    '''

    if classifier == 'knn':
        # K-nearest neighbours
        ks = np.arange(1, 5, 4)
        # ks = np.arange(3, 36, 4)
        for k in ks:
            clf = KNeighborsClassifier(k, weights='uniform')
    elif classifier == 'logistic':
        # Logistic Regression
        clf = LogisticRegression()

    elif classifier == 'svm':
        clf = svm.LinearSVC()
    else:
        raise ValueError

    clf.fit(train_x, train_y)
    guess_y = clf.predict(test_x)
    score = np.sum(guess_y == test_y) * 1.0 / len(guess_y)
    return score


def test_sk_methods():
    # Classify Kanade
    train, valid, test = loader.load_kanade(shared=False, set_name='sharp_equi25_25', pre={'scale2unit': True}, n=10000)
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test


    # K-nearest neighbours
    ks = np.arange(1, 6, 4)
    # ks = np.arange(3, 36, 4)
    for k in ks:
        clf = KNeighborsClassifier(k, weights='uniform')
        clf.fit(train_x, train_y)
        guess_y = clf.predict(test_x)
        print '{}-NN: {}'.format(k, np.sum(guess_y == test_y) * 1.0 / len(guess_y))

    # Logistic Regression
    clf = LogisticRegression()
    clf.fit(train_x, train_y)
    guess_y = clf.predict(test_x)
    print 'Logistic Regression: {}'.format(np.sum(guess_y == test_y) * 1.0 / len(guess_y))


    print 'SVM'
    # SVM
    clf = svm.SVC()
    clf.fit(train_x, train_y)
    guess_y = clf.predict(test_x)
    print 'SVC: {}'.format(np.sum(guess_y == test_y) * 1.0 / len(guess_y))

    # NuSVC
    # clf = svm.NuSVC()
    # clf.fit(train_x, train_y)
    # guess_y = clf.predict(test_x)
    # print 'NuSVC: {}'.format(np.sum(guess_y == test_y) * 1.0 / len(guess_y))

    # SVM
    clf = svm.LinearSVC()
    clf.fit(train_x, train_y)
    guess_y = clf.predict(test_x)
    print 'LinearSVC: {}'.format(np.sum(guess_y == test_y) * 1.0 / len(guess_y))

if __name__ == '__main__':
    test_sk_methods()