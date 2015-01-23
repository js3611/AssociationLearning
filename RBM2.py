import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from theano.tensor.shared_randomstreams import RandomStreams
import time

try:
    import PIL.Image as Image
except ImportError:
    import Image

import os

from utils import tile_raster_images
from utils import load_data

# Macro
t_float_x = theano.config.floatX

# Theano Debugging Configuration
# compute_test_value is 'off' by default, meaning this feature is inactive
# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class TrainParam(object):
    def __init__(self,
                 epochs=15,
                 batch_size=20,
                 n_chains=20,
                 n_samples=10,
                 plot_epoch_image=True,
                 plot_sample=True,
                 learning_rate=0.1,
                 momentum_type='nesterov',
                 momentum=0.5,
                 weight_decay=0.001,
                 sparsity_constraint=True,
                 sparsity_target=0.01,      # in range (0.1^9, 0.01)
                 sparsity_cost=0.01,
                 sparsity_decay=0.1
                 ):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_chains = n_chains
        self.n_samples = n_samples
        # Output Configuration
        self.plot_epoch_image = plot_epoch_image
        self.plot_sample = plot_sample
        # Weight Update Parameters
        self.learning_rate = learning_rate
        self.momentum_type = momentum_type
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Sparsity Constraint Parameters
        self.sparsity_constraint = sparsity_constraint
        self.sparsity_target = sparsity_target
        self.sparsity_cost = sparsity_cost
        self.sparsity_decay = sparsity_decay


class RBM(object):

    def __init__(self, 
                 v_n,
                 h_n,
                 W=None,
                 h_bias=None,
                 v_bias=None,
                 h_activationFnc=log_sig,
                 v_activationFnc=log_sig,
                 cd_type='persistent',
                 cd_steps=1,
                 train_parameters=None):

        np_rand = np.random.RandomState(1234)
        self.rand = RandomStreams(np_rand.randint(2 ** 30))

        if W is None:
            initial_W = np.asarray(
                np_rand.uniform(
                    low=-4 * np.sqrt(6. / (h_n + v_n)),
                    high=4 * np.sqrt(6. / (h_n + v_n)),
                    size=(v_n, h_n)
                ),
                dtype=t_float_x
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if v_bias is None:
            v_bias = theano.shared(
                value=np.zeros(v_n, dtype=t_float_x),
                name='v_bias', borrow=True
            )

        if h_bias is None:
            h_bias = theano.shared(
                value=np.zeros(h_n, dtype=t_float_x),
                name='h_bias', borrow=True
            )

        if train_parameters is None:
            train_parameters = TrainParam()

        self.train_parameters = train_parameters

        # Weights
        self.W = W
        # Visible Layer
        self.v_n = v_n
        self.v_bias = v_bias
        self.v_activationFnc = v_activationFnc
        # Hidden Layer
        self.h_n = h_n
        self.h_bias = h_bias
        self.h_activationFnc = h_activationFnc
        # Gibbs Sampling Method
        self.cd_type = cd_type
        self.cd_steps = cd_steps

        print "Initialised"

    def free_energy(self, v):        
        wv_c = T.dot(v, self.W) + self.h_bias
        return - T.dot(v, self.v_bias) - T.sum(T.log(1 + T.exp(wv_c)))

    def prop_up(self, v):
        """Propagates v to the hidden layer. """
        h_total_input = T.dot(v, self.W) + self.h_bias
        h_p_activation = self.h_activationFnc(h_total_input)
        return [h_total_input, h_p_activation]

    def prop_down(self, h):
        """Propagates h to the visible layer. """
        v_total_input = T.dot(h, self.W.T) + self.v_bias
        v_p_activation = self.v_activationFnc(v_total_input)
        return [v_total_input, v_p_activation]

    def sample_h_given_v(self, v):
        h_total_input, h_p_activation = self.prop_up(v)
        h_sample = self.rand.binomial(size=h_p_activation.shape,
                                      n=1,
                                      p=h_p_activation,
                                      dtype=t_float_x)
        return [h_total_input, h_p_activation, h_sample]

    def sample_v_given_h(self, h_sample):
        v_total_input, v_p_activation = self.prop_down(h_sample)
        v_sample = self.rand.binomial(size=v_p_activation.shape,
                                      n=1,
                                      p=v_p_activation,
                                      dtype=t_float_x)
        return [v_total_input, v_p_activation, v_sample]

    def gibbs_hvh(self, h):
        v_total_input, v_p_activation, v_sample = self.sample_v_given_h(h)
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v_sample)
        return [v_total_input, v_p_activation, v_sample,
                h_total_input, h_p_activation, h_sample]

    def gibbs_vhv(self, v):
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v)
        v_total_input, v_p_activation, v_sample = self.sample_v_given_h(h_sample)
        return [h_total_input, h_p_activation, h_sample,
                v_total_input, v_p_activation, v_sample]

    def contrastive_divergence(self, k=1):
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(self.input)
        chain_start = h_sample
        (
            [
                v_total_inputs,
                v_p_activations,
                v_samples,
                h_total_inputs,
                h_p_activations,
                h_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        chain_end = v_samples[-1]

        # Return result of scan
        return [updates,
                chain_end,
                v_total_inputs,
                v_p_activations,
                v_samples,
                h_total_inputs,
                h_p_activations,
                h_samples]

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        """Approximation to the reconstruction error (denoised auto encoder -> read up on this later"""

        cross_entropy = T.mean(
            T.sum(
                self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )

        return cross_entropy

    def train(self, input):
        print "Training"
        # TODO write train function
        return None

    def retrieve_parameters(self):
        # TODO
        return None