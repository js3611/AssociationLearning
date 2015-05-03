import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
from utils import load_data

import logistic_sgd

import os
import time
import datastorage
try:
    import PIL.Image as Image
except ImportError:
    import Image


CLASSICAL = 'classical'
NESTEROV = "nesterov"
PERSISTENT = "persistent"

# Macro
t_float_x = theano.config.floatX

# Path
root_dir = os.getcwd()
data_dir = "/".join([root_dir, "data"])


# Theano Debugging Configuration
# compute_test_value is 'off' by default, meaning this feature is inactive
# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class TrainParam(object):
    def __init__(self,
                 epochs=15,
                 batch_size=20,
                 plot_during_training=True,
                 output_directory=None,
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
        # Output Configuration
        self.output_directory = output_directory
        self.plot_during_training = plot_during_training

    def __str__(self):
        return "epoch" + str(self.epochs) + \
               "_batch" + str(self.batch_size) + \
               "_lr" + str(self.learning_rate) + \
               "_" + self.momentum_type + str(self.momentum) + \
               "_wd" + str(self.weight_decay) + \
               ("_sparsity"
                + "_t" + str(self.sparsity_target)
                + "_c" + str(self.sparsity_cost) +
                "_d" + str(self.sparsity_decay) if self.sparsity_constraint else "")


class RBM(object):

    def __init__(self,
                 v_n,
                 v_n2,
                 h_n,
                 W=None,
                 U=None,
                 h_bias=None,
                 v_bias=None,
                 v_bias2=None,
                 h_activation_fn=log_sig,
                 v_activation_fn=log_sig,
                 v_activation_fn2=log_sig,
                 cd_type=PERSISTENT,
                 cd_steps=1,
                 train_parameters=None):

        np_rand = np.random.RandomState(123)
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

        if U is None:
            initial_U = np.asarray(
                np_rand.uniform(
                    low=-4 * np.sqrt(6. / (h_n + v_n2)),
                    high=4 * np.sqrt(6. / (h_n + v_n2)),
                    size=(v_n2, h_n)
                ),
                dtype=t_float_x
            )
            U = theano.shared(value=initial_U, name='U', borrow=True)

        if v_bias is None:
            v_bias = theano.shared(
                value=np.zeros(v_n, dtype=t_float_x),
                name='v_bias', borrow=True
            )

        if v_bias2 is None:
            v_bias2 = theano.shared(
                value=np.zeros(v_n2, dtype=t_float_x),
                name='v_bias2', borrow=True
            )

        if h_bias is None:
            h_bias = theano.shared(
                value=np.zeros(h_n, dtype=t_float_x),
                name='h_bias', borrow=True
            )

        if train_parameters is None:
            train_parameters = TrainParam()

        persistent = theano.shared(value=np.zeros((train_parameters.batch_size, h_n),
                                                  dtype=t_float_x), name="persistent")

        self.train_parameters = train_parameters

        # Weights
        self.W = W
        self.U = U
        # Visible Layer 1
        self.v_n = v_n
        self.v_bias = v_bias
        self.v_activation_fn = v_activation_fn
        # Visible Layer 2
        self.v_n2 = v_n2
        self.v_bias2 = v_bias2
        self.v_activation_fn2 = v_activation_fn2
        # Hidden Layer
        self.h_n = h_n
        self.h_bias = h_bias
        self.h_activation_fn = h_activation_fn
        # Gibbs Sampling Method
        self.cd_type = cd_type
        self.cd_steps = cd_steps
        self.persistent = persistent
        self.params = [self.W, self.v_bias, self.h_bias, self.U, self.v_bias2]

        print "... initialised RBM"

    def __str__(self):
        return "rbm_" + str(self.h_n) + \
               "_" + self.cd_type + str(self.cd_steps) + \
               "_" + str(self.train_parameters)

    def free_energy(self, v, v2=None, w=None, u=None, v_bias=None, v_bias2=None, h_bias=None):
        if not w:
            w = self.W
        if not v_bias:
            v_bias = self.v_bias
        if not h_bias:
            h_bias = self.h_bias

        if not v2:
            wv_c = T.dot(v, w) + h_bias
            return - T.dot(v, v_bias) - T.sum(T.log(1 + T.exp(wv_c)))

        # Associative - contribution from v2
        if not u:
            u = self.U
        if not v_bias2:
            v_bias2 = self.v_bias2
            return - T.dot(v, v_bias) - T.dot(v2, v_bias2) - T.sum(
                T.log(1 + T.exp(h_bias + T.dot(v2, u) + T.dot(v, w))))

    def prop_up(self, v, v2=None):
        """Propagates v to the hidden layer. """
        h_total_input = T.dot(v, self.W) + self.h_bias
        # Associative
        if np.any(v2):
            h_total_input += T.dot(v2, self.U)
        h_p_activation = self.h_activation_fn(h_total_input)
        return [h_total_input, h_p_activation]

    def __prop_down(self, h, connectivity, bias, activation_fn):
        """Propagates h to the visible layer. """
        v_total_input = T.dot(h, connectivity.T) + bias
        v_p_activation = activation_fn(v_total_input)
        return [v_total_input, v_p_activation]

    def prop_down(self, h):
        return self.__prop_down(h, self.W, self.v_bias, self.v_activation_fn)

    def prop_down_assoc(self, h):
        return self.__prop_down(h, self.U, self.v_bias2, self.v_activation_fn2)

    def sample_h_given_v(self, v, v2=None):
        h_total_input, h_p_activation = self.prop_up(v, v2)
        h_sample = self.rand.binomial(size=h_p_activation.shape,
                                      n=1,
                                      p=h_p_activation,
                                      dtype=t_float_x)
        return [h_total_input, h_p_activation, h_sample]

    def __sample_v_given_h(self, h_sample, prop_down_fn):
        v_total_input, v_p_activation = prop_down_fn(h_sample)
        v_sample = self.rand.binomial(size=v_p_activation.shape,
                                      n=1,
                                      p=v_p_activation,
                                      dtype=t_float_x)
        return [v_total_input, v_p_activation, v_sample]

## REFACTOR ALL THIS SHIT

    def sample_v_given_h(self, h_sample):
        return self.__sample_v_given_h(h_sample, self.prop_down)

    def sample_v_given_h_assoc(self, h_sample):
        return self.__sample_v_given_h(h_sample, self.prop_down) + self.__sample_v_given_h(h_sample, self.prop_down_assoc)

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

    def gibbs_hvh_assoc(self, h):
        v_total_input, v_p_activation, v_sample, v2_total_input, v2_p_activation, v2_sample = self.sample_v_given_h_assoc(h)
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v_sample, v2_sample)
        return [v_total_input, v_p_activation, v_sample,
                v2_total_input, v2_p_activation, v2_sample,
                h_total_input, h_p_activation, h_sample]

    # For getting y's
    def gibbs_hvh_fixed(self, h, x):
        v_total_input, v_p_activation, v_sample, v2_total_input, v2_p_activation, v2_sample = self.sample_v_given_h_assoc(h)
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(x, v2_sample)
        return [v_total_input, v_p_activation, v_sample,
                v2_total_input, v2_p_activation, v2_sample,
                h_total_input, h_p_activation, h_sample]

    def gibbs_vhv_assoc(self, v):
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v)
        v_total_input, v_p_activation, v_sample = self.sample_v_given_h(h_sample)
        return [h_total_input, h_p_activation, h_sample,
                v_total_input, v_p_activation, v_sample]

    def pcd(self, k=1):
        chain_start = self.persistent
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

        updates[self.persistent] = h_samples[-1]

        # Return result of scan
        return [updates,
                chain_end,
                v_total_inputs,
                v_p_activations,
                v_samples,
                h_total_inputs,
                h_p_activations,
                h_samples]

    def contrastive_divergence(self, x, k=1):
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(x)
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

    def contrastive_divergence_assoc(self, x, y, k=1):
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(x, y)
        chain_start = h_sample
        (
            [
                v_total_inputs,
                v_p_activations,
                v_samples,
                v2_total_inputs,
                v2_p_activations,
                v2_samples,
                h_total_inputs,
                h_p_activations,
                h_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh_assoc,
            outputs_info=[None, None, None, None, None,
                          None, None, None, chain_start],
            n_steps=k
        )
        chain_end = v_samples[-1]
        chain_end2 = v2_samples[-1]

        # Return result of scan
        return [updates,
                chain_end,
                v_total_inputs,
                v_p_activations,
                v_samples,
                chain_end2,
                v2_total_inputs,
                v2_p_activations,
                v2_samples,
                h_total_inputs,
                h_p_activations,
                h_samples]

    def negative_statistics(self, x, y=None):
        if self.cd_type is PERSISTENT:
            return self.pcd(self.cd_steps)
        elif y:
            print "assoc cd"
            return self.contrastive_divergence_assoc(x, y, 1)
        else:
            return self.contrastive_divergence(x, self.cd_steps)

    def get_reconstruction_cost(self, x, v_total_inputs):
        """Used to monitor progress when using CD-k"""
        p = self.v_activation_fn(v_total_inputs)
        cross_entropy = T.mean(
            T.sum(
                x * T.log(p) + (1 - x) * T.log(1 - p),
                axis=1
            )
        )
        return cross_entropy

    def get_pseudo_likelihood(self, x, updates):
        """Stochastic approximation to the pseudo-likelihood.
        Used to Monitor progress when using PCD-k"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(x)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.v_n * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.v_n

        return cost

    def get_cost_updates(self, x, param_increments, y=None):
        param = self.train_parameters
        # Cast parameters
        lr = T.cast(param.learning_rate, dtype=t_float_x)
        m = T.cast(param.momentum, dtype=t_float_x)
        weight_decay = T.cast(param.weight_decay, dtype=t_float_x)

        # Declare parameter update variables
        #old_DW, old_Dvbias, old_Dhbias = param_increments[:-1]
        old_DW, old_Dvbias, old_Dhbias, old_DU, old_Dvbias2 = param_increments[:-1]

        pre_updates = []

        if param.momentum_type is NESTEROV:
            pre_updates.append((self.W, self.W + m * old_DW))
            pre_updates.append((self.h_bias, self.h_bias + m * old_Dhbias))
            pre_updates.append((self.v_bias, self.v_bias + m * old_Dvbias))

            # pre_updates.append((self.U, self.U + m * old_DU))
            # pre_updates.append((self.v_bias2, self.v2_bias + m * old_Dv2bias))

        # Perform Gibbs Sampling to generate negative statistics
        res = self.negative_statistics(x, y)
        updates = res[0]
        v_sample = res[1]
        v_total_inputs = res[2]

        ## ASSOC
        v2_sample = res[5]

        print v2_sample

        # Differentiate cost function w.r.t params to get gradients for param updates
        # cost = T.mean(self.free_energy(x)) - T.mean(self.free_energy(v_sample))
        # g_W, g_v, g_h = T.grad(cost, self.params, consider_constant=[v_sample])

        cost = T.mean(self.free_energy(x, y)) - T.mean(self.free_energy(v_sample, v2_sample))
        g_W, g_v, g_h, g_U, g_v2 = T.grad(cost, self.params, consider_constant=[v_sample, v2_sample])


        new_DW = m * old_DW - lr * g_W
        new_Dhbias = m * old_Dhbias - lr * g_h
        new_Dvbias = m * old_Dvbias - lr * g_v

        new_DU = m * old_DU - lr * g_U
        new_Dvbias2 = m * old_Dvbias2 - lr * g_v2


        if param.momentum_type is NESTEROV:
            # Nesterov update:
            # v_new = m * v_old - lr * Df(x_old + m*v_old)
            # x_new = x_old + v_new
            # <=> x_new = [x_old + m * v_old] - lr * Df([x_old + m * v_old])
            new_W = self.W - lr * g_W
            new_hbias = self.h_bias - lr * g_h
            new_vbias = self.v_bias - lr * g_v

            # Weight Decay
            new_W -= lr * weight_decay * (self.W - m * old_DW)
            new_hbias -= lr * weight_decay * (self.h_bias - m * old_Dhbias)
            new_vbias -= lr * weight_decay * (self.v_bias - m * old_Dvbias)
        else:
            # Classical Momentum from Sutskever, Hinton.
            # v_new = momentum * v_old + lr * grad_wrt_w
            # w_new = w_old + v_new
            new_W = self.W + new_DW
            new_hbias = self.h_bias + new_Dhbias
            new_vbias = self.v_bias + new_Dvbias

            new_U = self.U + new_DU
            new_vbias2 = self.v_bias2 + new_Dvbias2

            # Weight Decay
            new_W -= lr * weight_decay * self.W
            new_hbias -= lr * weight_decay * self.h_bias
            new_vbias -= lr * weight_decay * self.v_bias

            new_U -= lr * weight_decay * self.U
            new_vbias2 -= lr * weight_decay * self.v_bias2

        print param.sparsity_constraint

        # Sparsity
        if param.sparsity_constraint:
            active_probability_h = param_increments[-1]
            sparsity_target = param.sparsity_target
            sparsity_cost = param.sparsity_cost
            sparsity_decay_rate = param.sparsity_decay
            # 1. Compute actual probability of hidden unit being active, q
            # 1.1 Get q_current (mean probability that a unit is active in each mini-batch
            _, h_p_activation = self.prop_up(x)
            # q is the decaying average of mean active probability in each batch
            q = sparsity_decay_rate * active_probability_h + (1-sparsity_decay_rate) * T.mean(h_p_activation, axis=0)

            # 1.2 Update q_current = q for next iteration
            updates[active_probability_h] = q

            # 2. Define Sparsity Penalty Measure (dim = 1 x n_hidden)
            sparsity_penalty = T.nnet.binary_crossentropy(sparsity_target, q)

            # 3. Get the derivative
            if self.h_activation_fn is log_sig:     # if sigmoid
                d_sparsity = q - sparsity_target
            else:
                # Summation is a trick to differentiate element-wise
                # (as non relevant terms in sum vanish because they are constant w.r.t the partial derivatives)
                d_sparsity = T.grad(T.sum(sparsity_penalty), q)

            # Apply derivitive scaled by sparsity_cost to the weights
            # 1. use same quantity to adjust each weight
            # new_hbias -= lr * sparsity_cost * d_sparsity
            # new_W -= lr * sparsity_cost * d_sparsity
            # 2. multiply quantity by dq/dw (chain rule)
            chain_W, chain_h = T.grad(T.sum(q), [self.W, self.h_bias])
            new_hbias -= lr * sparsity_cost * d_sparsity * chain_h
            new_W -= lr * sparsity_cost * d_sparsity * chain_W

        # update parameters
        updates[self.W] = new_W
        updates[self.h_bias] = new_hbias
        updates[self.v_bias] = new_vbias

        updates[self.U] = new_U
        updates[self.v_bias2] = new_vbias2

        # update velocities
        updates[old_DW] = new_DW
        updates[old_Dhbias] = new_Dhbias
        updates[old_Dvbias] = new_Dvbias

        updates[old_DU] = new_DU
        updates[old_Dvbias2] = new_Dvbias2

        if self.cd_type is PERSISTENT:
            # cost = self.get_reconstruction_cost(x, v_total_inputs)
            measure_cost = self.get_pseudo_likelihood(x, updates)
        else:
            measure_cost = self.get_reconstruction_cost(x, v_total_inputs)

        return measure_cost, updates, pre_updates

    def get_train_fn(self, train_data, assoc_data, assoc=False):
        param = self.train_parameters
        batch_size = param.batch_size
        index = T.lscalar()
        x = T.matrix('x')
        y = T.matrix('y')

        # Initialise Variables used for training
        # For momentum
        old_DW = theano.shared(value=np.zeros(self.W.get_value().shape, dtype=t_float_x),
                               name='old_DW', borrow=True)
        old_Dvbias = theano.shared(value=np.zeros(self.v_n, dtype=t_float_x),
                                   name='old_Dvbias', borrow=True)
        old_Dhbias = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
                                   name='old_Dhbias', borrow=True)
        old_DU = theano.shared(value=np.zeros(self.U.get_value().shape, dtype=t_float_x),
                               name='old_DU', borrow=True)
        old_Dvbias2 = theano.shared(value=np.zeros(self.v_n2, dtype=t_float_x),
                                   name='old_Dvbias2', borrow=True)

        # For sparsity cost
        active_probability_h = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
                                             name="active_probability_h",
                                             borrow=True)

        # param_increments = [old_DW, old_Dvbias, old_Dhbias, active_probability_h]
        param_increments = [old_DW, old_Dvbias, old_Dhbias, old_DU, old_Dvbias2, active_probability_h]

        if not assoc:
            cross_entropy, updates, pre_updates = self.get_cost_updates(x, param_increments)
            pre_train = theano.function([], updates=pre_updates)
            train_rbm = theano.function(
                [index],
                cross_entropy,  # use cross entropy to keep track
                updates=updates,
                givens={
                    x: train_data[index * batch_size: (index + 1) * batch_size]
                },
                name='train_rbm'
            )
        else:
            print "assoc data"
            print assoc_data.get_value().shape
            cross_entropy, updates, pre_updates = self.get_cost_updates(x, param_increments, y)
            pre_train = theano.function([], updates=pre_updates)
            train_rbm = theano.function(
                [index],
                cross_entropy,  # use cross entropy to keep track
                updates=updates,
                givens={
                    x: train_data[index * batch_size: (index + 1) * batch_size],
                    y: assoc_data[index * batch_size: (index + 1) * batch_size]
                },
                name='train_rbm'
            )
            return train_rbm

        def train_fn(i):
            pre_train()
            return train_rbm(i)

        return train_fn

    def move_to_output_dir(self):
        # Move to data directory first
        if os.getcwd() != data_dir:
            if not os.path.isdir("data"):
                os.makedirs("data")
            print "... moved to " + data_dir
            os.chdir(data_dir)
            
        # Move to output dir 
        p = self.train_parameters
        rbm_name = p.output_directory
        if not rbm_name:
            rbm_name = str(self)
                
        if not os.path.isdir(rbm_name):
            os.makedirs(rbm_name)
        os.chdir(rbm_name)
        print "... moved to " + rbm_name

    def create_and_move_to_output_dir(self):
        """Create and move to output dir"""
        param = self.train_parameters
        # if os.path.isdir("data" + str(self)) or os.path.isdir("data" + param.output_directory):
        #     return

        if param.plot_during_training:
            if not os.path.isdir("data"):
                os.makedirs("data")
            os.chdir("data")
            if not param.output_directory:
                out_dir = str(self)
            else:
                out_dir = param.output_directory

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            os.chdir(out_dir)
            print "... moved to: " + out_dir

    def train(self, train_data, train_label, assoc=False):
        """Trains RBM. For now, input needs to be Theano matrix"""

        param = self.train_parameters
        batch_size = param.batch_size
        mini_batches = train_data.get_value(borrow=True).shape[0] / batch_size

        if not assoc:
            train_fn = self.get_train_fn(train_data, None)
        else:
            train_fn = self.get_train_fn(train_data, train_label, True)

        plotting_time = 0.
        start_time = time.clock()       # Measure training time
        for epoch in xrange(param.epochs):
            mean_cost = []
            for batch_index in xrange(mini_batches):
                mean_cost += [train_fn(batch_index)]

            print 'Epoch %d, cost is ' % epoch, np.mean(mean_cost)

            if param.plot_during_training:
                tile_shape = (self.h_n / 10 + 1, 10)
                plotting_start = time.clock()       # Measure plotting time
                image = Image.fromarray(
                    tile_raster_images(
                        X=self.W.get_value(borrow=True).T,
                        img_shape=(28, 28),
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                )
                image.save('epoch_%i.png' % epoch)
                plotting_stop = time.clock()
                plotting_time += (plotting_stop - plotting_start)

        end_time = time.clock()
        pre_training_time = (end_time - start_time) - plotting_time

        print ('Training took %f minutes' % (pre_training_time / 60.))

        return [mean_cost]

    def classify(self, data):

        # obtain from rbm

        # input x, get y out

        # use argmax

        return 0

    def plot_samples(self, test_data):
        n_chains = 20   # Number of Chains to perform Gibbs Sampling
        n_samples_from_chain = 10  # Number of samples to take from each chain

        test_set_size = test_data.get_value(borrow=True).shape[0]
        rand = np.random.RandomState(1234)
        test_input_index = rand.randint(test_set_size - n_chains)
        # Sample after 1000 steps of Gibbs Sampling each time
        plot_every = 1000
        persistent_vis_chain = theano.shared(
            np.asarray(
                test_data.get_value(borrow=True)[test_input_index:test_input_index + n_chains],
                dtype=theano.config.floatX
            )
        )

        # Expression which performs Gibbs Sampling
        (
            [
                presig_hids,
                hid_mfs,
                hid_samples,
                presig_vis,
                vis_mfs,
                vis_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_vhv,
            outputs_info=[None, None, None, None, None, persistent_vis_chain],
            n_steps=plot_every
        )

        updates.update({persistent_vis_chain: vis_samples[-1]})

        # Function that runs above Gibbs sampling
        sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                                    updates=updates, name='sample_fn')

        image_data = np.zeros(
            (29 * n_samples_from_chain + 1, 29 * n_chains - 1),
            dtype='uint8'
        )

        # Generate image by plotting the sample from the chain
        for i in xrange(n_samples_from_chain):
            vis_mf, vis_sample = sample_fn()
            print ' ... plotting sample ', i
            image_data[29 * i:29 * i + 28, :] = tile_raster_images(
                X=vis_mf,
                img_shape=(28, 28),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )

        # construct image
        image = Image.fromarray(image_data)
        image.save('samples.png')

    def save(self):
        datastorage.store_object(self)
        print "... saved RBM object to " + os.getcwd() + "/" + str(self)

    def reconstruct(self, x, y=None, k=1):
        if not y:
            y = self.rand.binomial(size=(x.get_value().shape[0], self.v_n2), n=1, p=0.5, dtype=t_float_x)

        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(x, y)
        chain_start = h_sample
        (
            [
                v_total_inputs,
                v_p_activations,
                v_samples,
                v2_total_inputs,
                v2_p_activations,
                v2_samples,
                h_total_inputs,
                h_p_activations,
                h_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh_fixed,
            outputs_info=[None, None, None, None, None,
                          None, None, None, chain_start],
            non_sequences=[x],
            n_steps=k
        )

        # get y
        chain_end = v2_samples[-1]

        sample_y = theano.function([], [chain_end, v2_samples, v2_p_activations], updates=updates)

        # Return result of scan
        return sample_y()


def test_rbm():
    print "Testing RBM"

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.01,
                    momentum_type=CLASSICAL,
                    momentum=0.5,
                    weight_decay=0.01,
                    sparsity_constraint=False,
                    sparsity_target=0.01,
                    sparsity_cost=0.01,
                    sparsity_decay=0.1,
                    plot_during_training=True)

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 10

    rbm = RBM(n_visible,
              n_visible,
              n_hidden,
              cd_type=CLASSICAL,
              cd_steps=1,
              train_parameters=tr)

    rbm.move_to_output_dir()

    # Train RBM
    rbm.train(train_set_x, train_set_x, assoc=False)

    # Test RBM
    rbm.plot_samples(test_set_x)

    # Store Parameters
    rbm.save()

    # Load RBM (test)
    loaded = datastorage.retrieve_object(str(rbm))
    if loaded:
        print "... loaded trained RBM"

    # Move back to root
    os.chdir(root_dir)
    print "moved to ... " + root_dir


def get_target_vector(x):
    xs = np.zeros(10, dtype=t_float_x)
    xs[x] = 1
    return xs


def test_rbm_association_with_label():
    print "Testing Associtive RBM with simple label"

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Reformat the train label 2 -> [0, 0, 1, ...., 0 ]

    new_train_set_y = np.matrix(map(lambda x: get_target_vector(x), train_set_y.eval()))
    new_train_set_y = theano.shared(new_train_set_y)

    # Combine the input
    # train_set_xy = T.concatenate([train_set_x, train_set_y], 1)
    # print train_set_xy
    # print train_set_xy.eval().shape

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.1,
                    momentum_type=CLASSICAL,
                    momentum=0.5,
                    weight_decay=0.01,
                    plot_during_training=True,
                    output_directory="AssociationLabelTest",
                    sparsity_constraint=False,
                    epochs=2)

    n_visible = train_set_x.get_value().shape[1]
    n_visible2 = 10
    n_hidden = 10

    rbm = RBM(n_visible,
              n_visible2,
              n_hidden,
              cd_type=CLASSICAL,
              cd_steps=1,
              train_parameters=tr)

    # Classification test - Reconstruct y through x
    x_in = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True),
            dtype=t_float_x
        )
    )

    y, ys = rbm.reconstruct(x_in, None)
    sol = test_set_y.eval()
    guess = [np.argmax(lab == 1) for lab in y]
    diff = np.count_nonzero(sol - guess)

    print diff
    print diff / float(test_set_y.eval().shape[0])


def test_rbm_association():
    print "Testing Associative RBM"

    # Even odd test
    testcases = 600
    k = 10

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.01,
                    momentum_type=CLASSICAL,
                    momentum=0.5,
                    weight_decay=0.01,
                    plot_during_training=True,
                    output_directory="AssociationTest",
                    sparsity_constraint=False,
                    epochs=15)

    n_visible = train_set_x.get_value().shape[1]
    n_visible2 = n_visible
    n_hidden = 10

    rbm = RBM(n_visible,
              n_visible2,
              n_hidden,
              cd_type=CLASSICAL,
              cd_steps=1,
              train_parameters=tr)

    # Find 1 example which train_set_x[i] represents 0 and 1
    zero_idx = np.where(train_set_y.eval() == 0)[0]
    one_idx = np.where(train_set_y.eval() == 1)[0]
    zero_image = train_set_x.get_value(borrow=True)[zero_idx[0] - 1]
    one_image = train_set_x.get_value(borrow=True)[one_idx[0] - 1]

    # Repeat and take first 50000
    def f(x):
        return zero_image if x % 2 == 0 else one_image
    new_train_set_y = theano.shared(
        np.matrix(map(f, train_set_y.eval())),
        name="train_set_y"
    )

    # print train_set_x.get_value().shape
    # print new_train_set_y.get_value().shape
    # print new_train_set_y
    # print train_set_x

    # Load RBM (test)
    rbm.move_to_output_dir()
    loaded = datastorage.retrieve_object(str(rbm))
    if not loaded:
        # Train RBM - learn joint distribution
        rbm.train(train_set_x, new_train_set_y, True)
        rbm.save()
    else:
        rbm = loaded
        print "... loaded"

    # Reconstruct y through x
    x_in = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[1:testcases],
            # test_set_x.get_value(borrow=True),
            dtype=t_float_x
        )
    )

    print "... reconstruction of associated images"
    reconstructed_y, ys, reconstruction = rbm.reconstruct(x_in, None, k)

    print "... reconstructed"

    # Create Dataset to feed into logistic regression

    # Train data: get only 0's and 1's
    ty = train_set_y.eval()
    zero_ones = (ty == 0) | (ty == 1)  # Get indices which the label is 0 or 1
    train_x = theano.shared(train_set_x.eval()[zero_ones])
    train_y = theano.shared(ty[zero_ones])

    # Validation setL get only 0's and 1's
    ty = valid_set_y.eval()
    zero_ones = (ty == 0) | (ty == 1)
    valid_x = theano.shared(valid_set_x.eval()[zero_ones])
    valid_y = theano.shared(ty[zero_ones])

    # Test set: reconstructed y's become the input. Get the corresponding x's and y's
    test_x = theano.shared(reconstructed_y)
    test_y = theano.shared(np.array(map(lambda x: x % 2, train_set_y.eval()), dtype=np.int32))[1:testcases]

    #
    for i in xrange(0, k):
        tile_shape = (testcases / 10 + 1, 10)
        image = Image.fromarray(
            tile_raster_images(
                # X=self.W.get_value(borrow=True).T,
                # X=test_x.get_value(borrow=True),
                X=reconstruction[i],
                img_shape=(28, 28),
                tile_shape=tile_shape,
                tile_spacing=(1, 1)
            )
        )
        image.save('reconstructions_%i.png' % i)

    dataset = ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))

    # Classify the reconstructions
    logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset, 600)

    # Move back to root
    os.chdir(root_dir)
    print "moved to ... " + root_dir


if __name__ == '__main__':
    # test_rbm_association()
    test_rbm()