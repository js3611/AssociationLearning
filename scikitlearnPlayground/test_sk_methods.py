__author__ = 'joschlemper'

import utils
import kanade_loader as loader
import sklearn as ss
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from simple_classifiers import SimpleClassifier

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
    train, valid, test = loader.load_kanade(shared=False, set_name='sharp_equi25_25', pre={'scale2unit': True})
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


def test_classifier():
    # Classify Kanade
    train, valid, test = loader.load_kanade(shared=False, set_name='sharp_equi25_25', pre={'scale2unit': True})
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test

    clf = SimpleClassifier(classifier='knn', train_x=train_x, train_y=train_y)
    print 'KNN: {}'.format(clf.get_score(test_x, test_y))

    clf = SimpleClassifier(classifier='logistic', train_x=train_x, train_y=train_y)
    print 'Logistic Regression: {}'.format(clf.get_score(test_x, test_y))

    clf = SimpleClassifier(classifier='svm', train_x=train_x, train_y=train_y)
    print 'LinearSVC: {}'.format(clf.get_score(test_x, test_y))

    train, valid, test = loader.load_kanade(shared=True, pre={'scale2unit': True})
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test

    clf = SimpleClassifier(classifier='knn', train_x=train_x, train_y=train_y)
    print 'KNN: {}'.format(clf.get_score(test_x, test_y))

    clf = SimpleClassifier(classifier='logistic', train_x=train_x, train_y=train_y)
    print 'Logistic Regression: {}'.format(clf.get_score(test_x, test_y))

    clf = SimpleClassifier(classifier='svm', train_x=train_x, train_y=train_y)
    print 'LinearSVC: {}'.format(clf.get_score(test_x, test_y))


if __name__ == '__main__':
    test_sk_methods()
    test_classifier()