import os
import sys
import time

import numpy as np
import datastorage as store
import mnist_loader as loader

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import sgd_optimization_mnist as LogisticRegression
from mlp import HiddenLayer
from rbm import *
from dbn import DBN

try:
    import PIL.Image as Image
except ImportError:
    import Image

from utils import tile_raster_images
from utils import load_data
from utils import save_digits

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class DefaultADBNConfig(object):
    def __init__(self):
        n_visible = 784
        n_visible2 = n_visible
        n_association = 300

        self.topology_left = [n_visible, 100, 100]
        self.topology_right = [n_visible2, 100, 100]
        self.n_association = n_association
        self.reuse_dbn = True
        self.top_cd_type = 'classical'

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
        self.n_association = n_association



class AssociativeDBN(object):

    def __init__(self, config=DefaultADBNConfig(), data_manager=None):
        self.config = config
        self.data_manager=data_manager

        self.dbn_left = DBN(topology=config.topology_left,
                            tr=config.base_rbm_params,
                            out_dir='left',
                            data_manager=data_manager)
        if config.reuse_dbn:
            self.dbn_right = self.dbn_left
        else:
            self.dbn_right = DBN(topology=config.topology_right,
                                 tr=config.base_rbm_params,
                                 out_dir='right',
                                 data_manager=data_manager)
        n_top_left = config.topology_left[-1]
        n_top_right = config.topology_right[-1]

        ass_rbm = RBM(n_top_left,
                      n_top_right,
                      config.n_association,
                      associative=True,
                      cd_type=config.top_cd_type,
                      cd_steps=1,
                      train_parameters=config.top_rbm_params,
                      progress_logger=AssociationProgressLogger())

        self.association_layer = ass_rbm
        print 'top layer = {}'.format(str(ass_rbm))

    def train(self, x1, x2, cache=False, optimise=False):
        self.dbn_left.pretrain(x1, cache=cache, optimise=optimise)

        if self.config.reuse_dbn:
            self.dbn_right = self.dbn_left
        else:
            self.dbn_right.pretrain(x2, cache=cache, optimise=optimise)

        x1_np = self.dbn_left.bottom_up_pass(x1.get_value(True))
        x2_np = self.dbn_right.bottom_up_pass(x2.get_value(True))

        x1_features = theano.shared(x1_np)
        x2_features = theano.shared(x2_np)

        out_dir = 'association_layer/' + str(len(self.dbn_left.rbm_layers)) + '_' + str(len(self.dbn_right.rbm_layers)) + '/'
        load = self.data_manager.retrieve(str(self.association_layer),out_dir=out_dir)
        if load and cache:
            self.association_layer = load
        else:
            self.association_layer.train(x1_features, x2_features)
            self.data_manager.persist(self.association_layer, out_dir=out_dir)

    # TODO clean up input and output of each function (i.e. they should all return theano or optional flag)
    def recall(self, x, associate_steps=10, recall_steps=5, img_name='default'):
        ''' left dbn bottom-up -> associate -> right dbn top-down
        :param x: data
        :param associate_steps: top level gibbs sampling steps
        :param recall_steps: right dbn sampling
        :return:
        '''
        self.data_manager.move_to('reconstruct')
        print '... moved to {}'.format(os.getcwd())

        left = self.dbn_left
        top = self.association_layer
        right = self.dbn_right

        if utils.isSharedType(x):
            x = x.get_value(borrow=True)

        # Pass to association layer
        top_out = left.bottom_up_pass(x)
        assoc_in = theano.shared(top_out, 'top_in', allow_downcast=True)

        # Sample from the association layer
        associate_x = top.reconstruct_association(assoc_in, k=associate_steps)


        if recall_steps > 0:
            top_in = theano.shared(associate_x, 'associate_x', allow_downcast=True)
            # Allow right dbn to day dream by extracting top layer rbm
            right_top_rbm = right.rbm_layers[-1]
            ass, ass_p, ass_s = right_top_rbm.sample_v_given_h(top_in)
            associate_x_in = theano.function([], ass_s)()
            associate_x_reconstruct = right_top_rbm.reconstruct(associate_x_in, k=recall_steps)

            # pass down to visible units, take the penultimate layer because we sampled at the top layer
            if len(right.rbm_layers) > 1:
                res = right.top_down_pass(associate_x_reconstruct, start=len(right.rbm_layers)-1)
            else:
                res = associate_x_reconstruct
            # res = result.get_value(borrow=True)
        else:
            res = right.top_down_pass(associate_x)

        n = res.shape[0]
        save_digits(x, img_name+'original.png', shape=(n / 10, 10), img_shape=(25, 25))
        save_digits(res, img_name+'dbn_reconstruction.png', shape=(n / 10, 10), img_shape=(25, 25))

        self.data_manager.move_to_project_root()

        return res


def test_associative_dbn(i=0):
    print "Testing Associative DBN which tries to learn even-odd of numbers"

    # load dataset
    train, valid, test = loader.load_digits(n=[500, 100, 100], pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = loader.sample_image(train_y)

    # project set up
    project_name = 'AssociationDBNTest/{}'.format(i)
    data_manager = store.StorageManager(project_name)
    cache = True

    # initialise AssociativeDBN
    config = DefaultADBNConfig()
    config.reuse_dbn = False
    config.topology_left = [784, 100, 100]
    config.topology_right = [784, 100]
    config.n_association = 500
    associative_dbn = AssociativeDBN(config=config, data_manager=data_manager)

    save_digits(train_x.get_value(borrow=True)[1:100], 'n_orig.png',(10, 10))
    save_digits(train_x01.get_value(borrow=True)[1:100], 'n_ass.png',(10, 10))

    # Train RBM - learn joint distribution
    associative_dbn.train(train_x, train_x01, cache=True)
    print "... trained associative DBN"

    # Reconstruct images
    reconstructed_y = associative_dbn.recall(test_x, associate_steps=1, recall_steps=10)
    print "... reconstructed images"

    # Create Dataset to feed into logistic regression
    # Test set: reconstructed y's become the input. Get the corresponding x's and y's
    dataset01 = loader.load_digits(n=[1000, 100, 100], digits=[0, 1])
    dataset01[2] = (theano.shared(reconstructed_y), test_y)

    # Classify the reconstructions
    # TODO use sklearn to obtain accuracy/precision etc
    score = LogisticRegression(0.13, 100, dataset01, 100)

    print 'Score: {}'.format(str(score))




if __name__ == '__main__':
    test_associative_dbn()