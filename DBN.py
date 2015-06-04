import os
import sys
import time
import copy
import datastorage as store

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM
from rbm_config import *
from rbm_logger import *


class DBNConfig(object):
    def __init__(self,
                 numpy_rng=None,
                 theano_rng=None,
                 topology=[625, 100, 100],
                 load_layer=None,
                 n_outs=10,
                 out_dir='dbn',  # useful for associative dbn
                 rbm_configs=RBMConfig(),
                 training_parameters=TrainParam(),
                 data_manager=None):
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.topology = topology
        self.load_layer = load_layer
        self.n_outs = n_outs
        self.data_manager = data_manager
        self.training_parameters = training_parameters
        self.rbm_configs = rbm_configs
        self.out_dir = out_dir


class DBN(object):
    def __init__(self, config=DBNConfig()):

        numpy_rng = config.numpy_rng
        theano_rng = config.theano_rng
        topology = config.topology
        load_layer = config.load_layer
        n_outs = config.n_outs
        data_manager = config.data_manager
        out_dir = config.out_dir
        tr = config.training_parameters
        rbm_configs = config.rbm_configs

        n_ins = topology[1]
        hidden_layers_sizes = topology[1:]

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.fine_tuned = False
        self.inference_layers = []
        self.generative_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.topology = topology
        self.out_dir = out_dir
        self.data_manager = data_manager

        assert self.n_layers > 0

        if not numpy_rng:
            numpy_rng = np.random.RandomState(123)
        else:
            numpy_rng = np.random.RandomState(123)

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not (type(tr) is list):
            tr = [tr for i in xrange(self.n_layers)]

        if not (type(rbm_configs) is list):
            rbm_configs = [rbm_configs for i in xrange(self.n_layers)]


        # Create Layers
        self.x = T.matrix('x')
        self.y = T.ivector('y')  # labels presented as 1D vector
        for i in xrange(self.n_layers):
            # construct sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=topology[i],
                                        n_out=topology[i + 1],
                                        activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)


            rbm_config = rbm_configs[i]
            rbm_config.v_n = topology[i]
            rbm_config.h_n = topology[i + 1]
            # rbm_config.training_parameters = tr[i]  # Ensure it has parameters

            rbm_layer = RBM(rbm_config, W=sigmoid_layer.W, h_bias=sigmoid_layer.b)
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
        # index to a [mini]batch
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')

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

    def pretrain(self, train_data, cache=False, train_further=False, optimise=False):
        if type(cache) is not list:
            cache = np.repeat(cache, self.n_layers)
        if type(train_further) is not list:
            train_further= np.repeat(train_further, self.n_layers)

        layer_input = train_data
        for i in xrange(len(self.rbm_layers)):
            rbm = self.rbm_layers[i]
            print 'training layer {}, {}'.format(i, rbm)

            self.data_manager.move_to('{}/layer/{}/{}'.format(self.out_dir, i, rbm))

            # Check Cache
            cost = 0
            loaded = store.retrieve_object(str(rbm))
            if cache[i] and loaded:
                # TODO override neural network's weights too
                epochs = rbm.config.train_params.epochs
                rbm = loaded
                rbm.config.train_params.epochs = epochs
                # Override the reference
                self.rbm_layers[i] = rbm
                print "... loaded trained layer {}".format(rbm)

                if train_further[i]:
                    cost += np.mean(rbm.train(layer_input))
                    self.data_manager.persist(rbm)
            else:
                if rbm.train_parameters.sparsity_constraint:
                    rbm.set_initial_hidden_bias()
                    rbm.set_hidden_mean_activity(layer_input)
                cost += np.mean(rbm.train(layer_input))
                self.data_manager.persist(rbm)

            self.data_manager.move_to_project_root()
            # os.chdir('../..')

            # Pass the input through sampler method to get next layer input
            sampled_layer = rbm.sample_h_given_v(layer_input)
            transform_input = sampled_layer[2]
            f = theano.function([], transform_input)
            res = f()
            layer_input = theano.shared(res)
        return cost


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

    def bottom_up_pass(self, x, start=0, end=sys.maxint):
        '''
        From visible layer to top layer
        :param x: numpy input
        :param start: start_layer
        :param end: end_layer (default = end)
        :return:
        '''
        n_layer = len(self.rbm_layers)
        end = min(end, n_layer)
        assert (0 <= start < end <= n_layer)

        layer_input = T.matrix('x')
        chain_next = layer_input
        for i in xrange(start, end):
            rbm = self.rbm_layers[i]
            h, hp, hs = rbm.sample_h_given_v(chain_next)
            chain_next = hs
            # layer_input = vp

        gibbs_sampling = theano.function([layer_input], [h, hp, hs])
        h, hp, hs = gibbs_sampling(x)
        # For final layer, take the probability vp
        return hs

    def top_down_pass(self, x, start=sys.maxint, end=0):
        '''
        From top 2-layer to visible layer
        :param x:
        :return:
        '''
        n_layer = len(self.rbm_layers)
        start = min(start, n_layer)
        assert (0 <= end < start <= n_layer)

        layer_input = T.matrix('x')
        chain_next = layer_input
        for i in reversed(xrange(end, start)):
            rbm = self.rbm_layers[i]
            v, vp, vs = rbm.sample_v_given_h(chain_next)
            chain_next = vs
            # layer_input = vp

        gibbs_sampling = theano.function([layer_input], [v, vp, vs])
        v, vp, vs = gibbs_sampling(x)
        # For final layer, take the probability vp
        return vp

    def reconstruct(self, x, k=1, plot_n=None, plot_every=1, img_name='reconstruction.png'):
        '''
        Reconstruct image given cd-k
        - data: theano
        '''
        if utils.isSharedType(x):
            x = x.get_value(borrow=True)

        orig = x
        if self.n_layers > 1:
            top_x = self.bottom_up_pass(x, 0, self.n_layers-1)
        else:
            top_x = x
        # get top layer rbm
        rbm = self.rbm_layers[-1]

        # Set the initial chain
        chain_state = theano.shared(np.asarray(top_x,
                                               dtype=theano.config.floatX),
                                    name='reconstruct_root')

         # Gibbs sampling
        k_batch = k / plot_every
        (res, updates) = theano.scan(rbm.gibbs_vhv,
                                     outputs_info=[None, None, None,
                                                   None, None, chain_state],
                                     n_steps=plot_every,
                                     name="Gibbs_sampling_reconstruction")
        updates.update({chain_state: res[-1][-1]})
        gibbs_sampling = theano.function([], res, updates=updates)

        reconstructions = []
        for i in xrange(k_batch):
            result = gibbs_sampling()
            [_, _, _, _, reconstruction_chain, _] = result
            if self.n_layers > 1:
                recon = self.top_down_pass(reconstruction_chain[-1], self.n_layers-1)
            else:
                recon = reconstruction_chain[-1]
            reconstructions.append(recon)

        if self.rbm_layers[0].track_progress:
            self.rbm_layers[0].track_progress.visualise_reconstructions(orig,
                                                                        reconstructions,
                                                                        plot_n,
                                                                        img_name)

        return reconstructions[-1]

    def sample(self, n=1, k=10, rand_type='normal'):

        top_layer = self.rbm_layers[-1]

        if self.n_layers > 1:
            print 'sampling according to active_probability of layer below'
            pen = self.rbm_layers[-2]
            active_h = pen.active_probability_h.get_value(borrow=True)
            x = top_layer.sample(n, k, rand_type='binomial', p=active_h)
        else:
            # Sample between top two layers
            x = top_layer.sample(n, k, rand_type='binomial', p=0.05)

        # prop down the output to visible unit if it is not RBM
        if self.n_layers > 1:
            sampled = self.top_down_pass(x, self.n_layers - 1)
        else:
            sampled = x

        return sampled

    def untie_weights(self):
        layers = self.rbm_layers
        for rbm in layers:
            W = rbm.W.get_value(borrow=False)
            h_bias = rbm.h_bias.get_value(borrow=False)
            v_bias = rbm.v_bias.get_value(borrow=False)
            self.inference_layers.append(RBM(config=rbm.config, W=W, h_bias=h_bias, v_bias=v_bias))
            self.generative_layers.append(RBM(config=rbm.config, W=copy.deepcopy(W), h_bias=copy.deepcopy(h_bias), v_bias=copy.deepcopy(v_bias)))

    def fine_tune(self, x):
        '''
        Fine tunes DBN. Inference weights and Generative weights will be untied
        :param x:
        :return:
        '''

        self.untie_weights()

