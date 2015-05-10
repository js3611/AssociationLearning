import os
import sys
import time
import datastorage as store

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM, TrainParam, ProgressLogger


class DBN(object):
    
    def __init__(self, 
                 numpy_rng=None,
                 theano_rng=None,
                 topology=[784, 500, 500],
                 load_layer=None,
                 n_outs=10,
                 out_dir='dbn',
                 tr=TrainParam()):

        n_ins = topology[1]
        hidden_layers_sizes = topology[1:]

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.topology = topology
        self.out_dir = out_dir

        assert self.n_layers > 0

        if not numpy_rng:
            numpy_rng = np.random.RandomState(123)
        else:
            numpy_rng = np.random.RandomState(123)


        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # Create Layers
        self.x = T.matrix('x')
        self.y = T.ivector('y')  #labels presented as 1D vector
        for i in xrange(self.n_layers):
            # construct sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                         input = layer_input,
                                         n_in = topology[i],
                                         n_out = topology[i+1],
                                         activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # construct RBM that shared weights with this layer
            rbm_layer = RBM(associative=False,
                            cd_type='classical',
                            v_n=topology[i],
                            h_n=topology[i+1],
                            W=sigmoid_layer.W,
                            h_bias=sigmoid_layer.b,
                            train_parameters=tr,
                            progress_logger=ProgressLogger())
            self.rbm_layers.append(rbm_layer)
            
        # Logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # cost for fine tuning
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # gradient wrt model parameters
        self.errors = self.logLayer.errors(self.y)

    def __str__(self):
        return 'dbn_l' + str(self.n_layers) + \
               '_' + '_'.join([str(i) for i in self.topology])

    def pretraining_functions(self, train_set_x, batch_size, k):
        #index to a [mini]batch
        index = T.lscalar('index')
        learning_rate=T.scalar('lr')

        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        for rbm in self.rbm_layers:

#            cost, updates = rbm.get_cost_updates(learning_rate, persistent=None, k=k)
            cost, updates = rbm.get_cost_updates(learning_rate, k=k)

            # input gets propagated through layer weights W i.e. W2(W1x+b1)+b2 etc
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretrain(self, train_data, cache=False):
        layer_input = train_data
        for i in xrange(len(self.rbm_layers)):
            rbm = self.rbm_layers[i]
            print 'training layer {}, {}'.format(str(i), str(rbm))
            store.move_to(self.out_dir + '/layer' + str(i) + '/' + str(rbm))

            # Check Cache
            loaded = store.retrieve_object(str(rbm))
            if cache and loaded:
                # TODO override neural network's weights too
                rbm = loaded
                # Override the reference
                self.rbm_layers[i] = rbm
                print "... loaded trained layer"
            else:
                rbm.train(layer_input)
                store.store_object(rbm)

            os.chdir('../..')

            # Pass the input through sampler method to get next layer input
            sampled_layer = rbm.sample_h_given_v(layer_input)
            transform_input = sampled_layer[2]
            f = theano.function([], transform_input)
            res = f()
            layer_input = theano.shared(res)


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def bottom_up_pass(self, x):
        pass

    def top_down_pass(self, x):
        '''
        From top 2-layer to visible layer
        :param x:
        :return:
        '''
        layer_input = T.matrix('x')
        chain_next = layer_input
        for rbm in reversed(self.rbm_layers[:-1]):
            v, vp, vs = rbm.sample_v_given_h(chain_next)
            chain_next = vs
            # layer_input = vp

        gibbs_sampling = theano.function([layer_input], [v, vp, vs])
        v, vp, vs = gibbs_sampling(x)
        # For final layer, take the probability vp
        return vp

    def sample(self, n=1, k=10):

        top_layer = self.rbm_layers[-1]

        # Sample between top two layers
        x = top_layer.sample(n, k)

        # prop down the output to visible unit
        sampled = self.top_down_pass(x)

        return sampled