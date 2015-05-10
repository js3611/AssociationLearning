import os
import sys
import time

import numpy as np
import datastorage as store
import mnist_loader as loader

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import *
from DBN import DBN

try:
    import PIL.Image as Image
except ImportError:
    import Image

from utils import tile_raster_images
from utils import load_data
from utils import save_digits

class DefaultADBNConfig(object):
    def __init__(self):
        n_visible = 784
        n_visible2 = n_visible
        n_top_left = 100
        n_top_right = 100
        n_association = 300

        self.topology_left = [n_visible, 100, n_top_left]
        self.topology_right = [n_visible2, 100, n_top_right]
        self.n_association = n_association
        self.reuse_dbn = True

        tr = TrainParam(learning_rate=0.01,
                        momentum_type=NESTEROV,
                        momentum=0.5,
                        weight_decay=0.01,
                        sparsity_constraint=False,
                        sparsity_target=0.01,
                        sparsity_cost=0.01,
                        sparsity_decay=0.1,
                        epochs=50)

        self.base_rbm_params = tr
        self.top_rbm_params = tr

        self.n_visible = n_visible
        self.n_visible2 = n_visible2
        self.n_top_left = n_top_left
        self.n_top_right = n_top_right
        self.n_association = n_association


class AssociativeDBN(object):

    def __init__(self, config=DefaultADBNConfig()):
        self.config = config

        self.dbn_left = DBN(config.topology_left, tr=config.base_rbm_params)
        if config.reuse_dbn:
            self.dbn_right = self.dbn_left
        else:
            self.dbn_right = DBN(config.topology_right, tr=config.base_rbm_params)

        ass_rbm = RBM(config.n_top_left,
                      config.n_top_right,
                      config.n_association,
                      associative=True,
                      cd_type=PERSISTENT,
                      cd_steps=1,
                      train_parameters=config.top_rbm_params,
                      progress_logger=AssociationProgressLogger())

        self.association_layer = ass_rbm

    def train(self, x1, x2):

        # check cache

        self.dbn_left.pretrain(x1)

        if self.config.reuse_dbn:
            self.dbn_right = self.dbn_left
        else:
            self.dbn_right.pretrain(x2)

        x1_features = self.dbn_left.bottom_up_pass(x1)
        x2_features = self.dbn_left.bottom_up_pass(x2)

        self.association_layer.train(x1_features, x2_features)

    def recall(self, x, associate_steps=3, recall_steps=5):
        ''' left dbn bottomup -> associate -> right dbn topdown
        :param x: data
        :param associate_steps: top level gibbs sampling steps
        :param recall_steps: right dbn sampling
        :return:
        '''

        left = self.dbn_left
        top = self.association_layer
        right = self.dbn_right

        # Pass to association layer
        left.bottom_up_pass(x)

        # Sample from the association layer
        associate_x = top.reconstruct_association(x, k=associate_steps)

        # Allow right dbn to day dream by extracting top layer rbm
        right_top_rbm = right.rbm_layers[-1]
        associate_x_in = right_top_rbm.sample_v_given_h(associate_x)
        associate_x_reconstruct = right_top_rbm.reconstruct(associate_x_in, k=recall_steps)

        # pass down to visible units
        return right.top_down_pass(associate_x_reconstruct)


def test_associative_dbn():
    print "Testing Associative DBN which tries to learn even-odd of numbers"
    cache = True

    train, valid, test = loader.load_digits(n=[500, 100, 100], digits=[0, 1, 2, 3], pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = loader.sample_image(train_y)

    dataset01 = loader.load_digits(n=[500, 100, 100], digits=[0, 1])
    train, valid, test = loader.load_digits(n=[500, 100, 100])

    store.move_to('AssociationDBNTest')

    associative_dbn = AssociativeDBN()

    # Train RBM - learn joint distribution
    associative_dbn.train(train_x, train_x01)

    print "... reconstruction of associated images"
    reconstructed_y = associative_dbn.recall(test_x)
    print "... reconstructed"

    # TODO use sklearn to obtain accuracy/precision etc

    # Create Dataset to feed into logistic regression
    # Test set: reconstructed y's become the input. Get the corresponding x's and y's
    dataset01[2] = (theano.shared(reconstructed_y), test_y)

    # Classify the reconstructions
    score = LogisticRegression.sgd_optimization_mnist(0.13, 100, dataset01, 100)

    print 'Score: {}'.format(str(score))




if __name__ == '__main__':
    test_associative_dbn()