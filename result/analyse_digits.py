__author__ = 'joschlemper'

import matplotlib.pyplot as plt
import re
import numpy as np
from kanade_loader import emotion_dict, emotion_rev_dict


def rename(name):
    return '-'.join(re.findall("\d+.\d+", name))

def rename_adbn(name):
    d = re.findall("\d+", name)
    str = 'L[{}-{}]-R[{}-{}]-{}'.format(d[0], d[1], d[2], d[3], d[4])
    return str


def dbn_digits(plot='pre'):
    minval = 1
    # Read file
    f = open('dbn_hidden_n.txt')
    lines = f.readlines()
    f.close()
    length = len(lines)
    stats = {}
    for line in lines:
        iter, name, scores = line.strip().split(':')
        stats[name] = []

    for line in lines:
        iter, name, scores = line.strip().split(':')
        vals = map(lambda x: 1 - float(x), scores.split(',')[:-1])
        if plot == 'Pre-Training':
            v = vals[0]
        else:
            v = np.min(vals[1:])
            # v = vals[1]

        stats[name].append(v)
        minval = min(min(vals),minval)

    keys = ['[1568, 250, 250]', '[1568, 250, 500]', '[1568, 500, 250]',
            '[1568, 500, 500]', '[1568, 500, 1000]', '[1568, 1000, 500]']

    for key in keys:
        vals = stats[key]
        plt.plot([x*10 for x in xrange(1, 11)], vals, label=rename(key))

    rbm_result = [0.72, 0.86, 0.87, 0.86, 0.87, 0.83, 0.81, 0.81, 0.86, 0.89, 0.88,
                  0.83, 0.84, 0.84, 0.85, 0.86, 0.84, 0.78, 0.80, 0.82, 0.62, 0.72,
                  0.69, 0.60, 0.61, 0.80, 0.56, 0.66, 0.60, 0.71, 0.63, 0.53, 0.55,
                  0.50, 0.56, 0.53, 0.53, 0.49, 0.53, 0.51, 0.50, 0.51, 0.51, 0.52,
                  0.51, 0.50, 0.51, 0.51, 0.50, 0.51] + [0.5]*50

    plt.ylim(0, 0.55)
    # plt.xticks(range(10), [10 + x * 10 for x in range(10)])
    plt.plot(range(10, 100), map(lambda x: 1 - float(x), rbm_result[10:]), 'k--', label='RBM',)


    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(loc=0)
    plt.title('Error rate vs Training Epochs (%s)' % plot)
    plt.show()
    print minval

def adbn_digits(plot='Pre-Training'):
    minval = 1
    # Read file
    f = open('adbn_errors.txt')
    lines = f.readlines()
    f.close()
    length = len(lines)
    stats = {}
    for line in lines:
        iter, name, scores = line.strip().split(':')
        stats[rename_adbn(name)] = []

    for line in lines:
        iter, name, scores = line.strip().split(':')
        vals = map(lambda x: 1 - float(x), scores.split(',')[:-1])
        if plot == 'Pre-Training':
            v = vals[0]
        else:
            # v = min(vals)
            v = min(vals[0:])

        stats[rename_adbn(name)].append(v)
        minval = min(min(vals),minval)

    keys = [
            # 'L[784-100]-R[784-50]-100',
            # 'L[784-250]-R[784-50]-100',
            # 'L[784-500]-R[784-50]-100',
            # 'L[784-100]-R[784-50]-250',
            'L[784-250]-R[784-50]-250',
            'L[784-500]-R[784-50]-250',
            # 'L[784-100]-R[784-50]-500',
            'L[784-250]-R[784-50]-500',
            'L[784-500]-R[784-50]-500',
            # 'L[784-100]-R[784-50]-1000',
            'L[784-250]-R[784-50]-1000',
            'L[784-500]-R[784-50]-1000']

    for key in keys:
            vals = stats[key]
            plt.plot(vals, label=key)

    plt.ylim(0, 0.5)
    plt.xticks(range(10), [10 + x * 10 for x in range(10)])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(loc=0)
    plt.title('Error rate vs Training Epochs (%s)' % plot)
    plt.show()
    print minval

def rbm_digits():
    global rbm_result, errors
    rbm_result = [0.72, 0.86, 0.87, 0.86, 0.87, 0.83, 0.81, 0.81, 0.86, 0.89, 0.88,
                  0.83, 0.84, 0.84, 0.85, 0.86, 0.84, 0.78, 0.80, 0.82, 0.62, 0.72,
                  0.69, 0.60, 0.61, 0.80, 0.56, 0.66, 0.60, 0.71, 0.63, 0.53, 0.55,
                  0.50, 0.56, 0.53, 0.53, 0.49, 0.53, 0.51, 0.50, 0.51, 0.51, 0.52,
                  0.51, 0.50, 0.51, 0.51, 0.50, 0.51]
    errors = map(lambda x: 1 - x, rbm_result)
    # from pylab import polyfit,poly1d
    # f = poly1d(polyfit(np.arange(0,len(errors)), errors, 100))
    # plt.plot(f(np.linspace(0,1000,50)), label='smooth')
    plt.plot(errors, label='RBM1568-332')
    plt.ylim(0, 0.7)
    # plt.xticks(range(10), [10 + x * 10 for x in range(10)])
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend(loc=0)
    plt.title('Error rate vs Training Epochs')
    plt.show()
    print 1 - max(rbm_result)


rbm_digits()
dbn_digits('Pre-Training')
dbn_digits('Fine-Tuned')
adbn_digits('Pre-Training')
adbn_digits('Fine-Tuned')

