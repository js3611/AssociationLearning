__author__ = 'joschlemper'

import numpy as np
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from activationFunction import *


class RBMUnit(object):
    '''
    By default implements stochastic binary unit
    '''
    def __init__(self, *args):
        self.np_rand = np.random.RandomState(123)
        self.rand = RandomStreams(self.np_rand.randint(2 ** 30))

    def scale(self, x):
        return log_sig(x)

    def activate(self, p_activate):
        x_sample = self.rand.binomial(size=p_activate.shape, n=1, p=p_activate, dtype=theano.config.floatX)
        return x_sample

    def energy(self, v, v_bias):
        return - T.dot(v, v_bias)


class BinaryUnit(RBMUnit):

    def activate(self, x):
        p_activate = log_sig(x)
        return p_activate


class GaussianVisibleUnit(RBMUnit):
    ''' Gaussian Visible Unit
    Note:
    From the layer above, the input is simply id(Wh + b)
    '''

    def activate(self, x):
        return x # id
        # return x + self.rand.normal(size=x.shape, avg=0., std=1., dtype=theano.config.floatX)
    def energy(self, v, v_bias):
        return 0.5 * T.sum((v - v_bias) ** 2)

    def scale(self, x):
        # return (x - T.mean(x, axis=0)) / T.std(x, axis=0)  # normalise
        return x



class ReLUnit(RBMUnit):
    '''
    Rectifier Linear Unit
    - underlying mathematics is the same as stochastic binary unit
    - activated by f(x) = max(0, x)
    '''
    def __init__(self, size):
        RBMUnit.__init__(self)
        self.size = size

    def scale(self, x):
        return x

    def activate(self, x):
        return T.maximum(0, x)

class NReLUnit(ReLUnit):
    ''' Noisy Rectified Linear Unit

    To sample from it, we approximate by f(x) = max(0, x + N(0, V)), V = 1 or V = std(x)

    '''
    def activate(self, x):
        return T.maximum(0, x + self.rand.normal(size=x.shape, avg=0., std=1., dtype=theano.config.floatX))
        # T.std(x, axis=0)
