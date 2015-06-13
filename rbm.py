import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from rbm_units import *
from rbm_logger import *
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import m_loader
import datastorage as store
import rbm_config

import sys
import os
import time
import math

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


class RBM(object):
    def __init__(self,
                 config,
                 W=None,
                 U=None,
                 h_bias=None,
                 v_bias=None,
                 v_bias2=None):

        cd_type = config.cd_type
        cd_steps = config.cd_steps
        associative = config.associative
        v_n = config.v_n
        v_n2 = config.v2_n
        h_n = config.h_n

        self.np_rand = np.random.RandomState(123)
        self.rand = RandomStreams(self.np_rand.randint(2 ** 30))

        W = self.get_initial_weight(W, v_n, h_n, 'W')
        v_bias = self.get_initial_bias(v_bias, v_n, 'v_bias')
        h_bias = self.get_initial_bias(h_bias, h_n, 'h_bias')

        train_params = config.train_params

        persistent = theano.shared(value=np.zeros((train_params.batch_size, h_n),
                                                  dtype=t_float_x), name="persistent")

        # For sparsity cost
        active_probability_h = theano.shared(value=np.zeros(h_n, dtype=t_float_x),
                                             name="active_probability_h")

        self.dropout = train_params.dropout
        self.dropout_rate = train_params.dropout_rate
        if self.dropout:
            self.dropout_mask = self.rand.binomial(size=(h_n,), p=self.dropout_rate, n=1)

        self.train_parameters = train_params

        # Weights
        self.W = W

        # Visible Layer
        self.v_n = v_n
        self.v_bias = v_bias
        self.v_unit = config.v_unit()

        # Hidden Layer
        self.h_n = h_n
        self.h_bias = h_bias
        self.h_unit = config.h_unit(self.h_n)
        # Gibbs Sampling Method
        self.cd_type = cd_type
        self.cd_steps = cd_steps
        self.persistent = persistent
        self.params = [self.W, self.v_bias, self.h_bias]

        # For hyperparameters
        self.active_probability_h = active_probability_h

        if associative:
            self.U = self.get_initial_weight(U, v_n2, h_n, 'U')
            # Visible Layer 2
            self.v_n2 = v_n2
            self.v_bias2 = self.get_initial_bias(v_bias2, v_n2, 'v_bias2')
            self.v_unit2 = config.v2_unit()
            self.params += [self.U, self.v_bias2]

        self.associative = associative
        self.track_progress = config.progress_logger
        self.config = config
        self.training = False

        # Check for legit configuration
        if train_params.sparsity_constraint and type(self.h_unit) is not RBMUnit:
            raise Exception('Sparsity Constraint can be used only for Stochastic Binary Hidden Unit')

        if train_params.sparsity_constraint:
            self.set_initial_hidden_bias()
            new_p_h = np.repeat(train_params.sparsity_target, h_n).astype(t_float_x)
            self.active_probability_h = theano.shared(value=new_p_h,
                                                      name="active_probability_h")

    def __str__(self):
        name = 'ass_' if self.associative else ''
        return name + "rbm_{}-{}_{}{}_{}".format(self.v_n, self.h_n, self.cd_type, self.cd_steps, self.train_parameters)

    def get_initial_weight(self, w, nrow, ncol, name):
        if w is None:
            w = np.asarray(
                self.np_rand.normal(0, scale=0.01, size=(nrow, ncol)),
                # self.np_rand.uniform(low=-1./10, high=1./10, size=(nrow, ncol)),
                # self.np_rand.uniform(
                # low=-4 * np.sqrt(6. / (nrow + ncol)),
                # high=4 * np.sqrt(6. / (nrow + ncol)),
                # size=(nrow, ncol)
                # ),
                dtype=t_float_x
            )
        if 'numpy' in str(type(w)):
            return theano.shared(value=w, name=name, borrow=True)
        else:
            return w

    def get_initial_bias(self, bias, n, name):
        if bias is None:
            bias = np.zeros(n, dtype=t_float_x)

        if 'numpy' in str(type(bias)):
            return theano.shared(value=bias, name=name, borrow=True)
        else:
            return bias

    def set_initial_visible_bias(self, train_data):
        '''
        Sets initial bias for visible unit i to
        log [pi / (1 - pi)], where pi = probability unit i is on (i.e. mean value of all training data)
        FOR BINARY STOCHASTIC UNITS ONLY
        :param train_data:
        :return:
        '''
        if type(self.v_unit) is RBMUnit:
            print '... Readjusting the initial visible bias (for stochastic binary unit)'
            p = theano.function([], T.mean(train_data, axis=0))()
            self.v_bias.set_value(np.log2(p / (1 - p)))

    def set_initial_hidden_bias(self):
        if self.train_parameters.sparsity_constraint:
            print '... Sparsity: setting initial bias for Stochastic Binary Hidden Unit'
            t = self.train_parameters.sparsity_target
            bias = np.log(t / (1 - t))
            self.h_bias.set_value(np.tile(bias, self.h_n).astype(t_float_x))

    def free_energy(self, v, v2=None):
        if self.associative:
            return self.calc_free_energy(v, v2)
        else:
            return self.calc_free_energy(v)

    def calc_free_energy(self, v, v2=None):
        w = self.W
        v_bias = self.v_bias
        h_bias = self.h_bias

        t0 = self.v_unit.energy(v, v_bias)
        t1 = T.dot(v, w) + h_bias

        if type(v2) is not type(None):
            u = self.U
            v_bias2 = self.v_bias2
            # t0 += -T.dot(v2, v_bias2) # For classRBM
            t0 += self.v_unit2.energy(v2, v_bias2)  # For GaussianUnits
            t1 += T.dot(v2, u)

        t2 = - T.sum(T.log(1 + T.exp(t1)))

        return t0 + t2

    def prop_up(self, v, v2=None):
        """Propagates v to the hidden layer. """
        h_total_input = T.dot(v, self.W) + self.h_bias

        if np.any(v2):  # Associative
            h_total_input += T.dot(v2, self.U)

        h_p_activation = self.h_unit.scale(h_total_input)
        if self.training and self.dropout:
            h_p_activation *= self.dropout_mask

        return [h_total_input, h_p_activation]

    def __prop_down(self, h, connectivity, bias, v_unit):
        """Propagates h to the visible layer. """
        v_in = T.dot(h, connectivity.T) + bias
        return [v_in, (v_unit.scale(v_in))]

    def prop_down(self, h):
        return self.__prop_down(h, self.W, self.v_bias, self.v_unit)

    def prop_down_assoc(self, h):
        return self.__prop_down(h, self.U, self.v_bias2, self.v_unit2)

    def sample_h_given_v(self, v, v2=None):
        h_total_input, h_p_activation = self.prop_up(v, v2)
        h_sample = self.h_unit.activate(h_p_activation)
        return [h_total_input, h_p_activation, h_sample]

    def __sample_v_given_h(self, h_sample, prop_down_fn):
        v_total_input, v_p_activation = prop_down_fn(h_sample)
        v_sample = self.v_unit.activate(v_p_activation)
        return [v_total_input, v_p_activation, v_sample]

    def sample_v_given_h(self, h_sample):
        return self.__sample_v_given_h(h_sample, self.prop_down)

    def sample_v_given_h_assoc(self, h_sample):
        return self.__sample_v_given_h(h_sample, self.prop_down) + self.__sample_v_given_h(h_sample,
                                                                                           self.prop_down_assoc)

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
        v_total_input, v_p_activation, v_sample, v2_total_input, v2_p_activation, v2_sample = self.sample_v_given_h_assoc(

            h)
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v_sample, v2_sample)
        return [v_total_input, v_p_activation, v_sample,
                v2_total_input, v2_p_activation, v2_sample,
                h_total_input, h_p_activation, h_sample]

    # For getting y's
    def gibbs_hvh_fixed(self, h, x):
        v_total_input, v_p_activation, v_sample, v2_total_input, v2_p_activation, v2_sample = self.sample_v_given_h_assoc(
            h)
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(x, v2_sample)
        return [v_total_input, v_p_activation, v_sample,
                v2_total_input, v2_p_activation, v2_sample,
                h_total_input, h_p_activation, h_sample]

    # For getting y's
    def gibbs_hvh_fixed2(self, h, x):
        v_total_input, v_p_activation, v_sample = self.sample_v_given_h(h)
        v_sample = T.concatenate([x, v_sample[:, (self.v_n / 2):]], axis=1)
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v_sample)
        return [v_total_input, v_p_activation, v_sample,
                h_total_input, h_p_activation, h_sample]

    def gibbs_vhv_assoc(self, v):
        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(v)
        v_total_input, v_p_activation, v_sample = self.sample_v_given_h(h_sample)
        return [h_total_input, h_p_activation, h_sample,
                v_total_input, v_p_activation, v_sample]

    def pcd_assoc(self, k=1):
        chain_start = self.persistent
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

        updates[self.persistent] = h_samples[-1]

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
        '''
        :param x:
        :param k:
        :return: chain_end is the last v visible
        '''

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
                h_samples,
                h_sample]

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
            if y:
                return self.pcd_assoc(self.cd_steps)
            else:
                return self.pcd(self.cd_steps)
        elif y:
            return self.contrastive_divergence_assoc(x, y, self.cd_steps)
        else:
            return self.contrastive_divergence(x, self.cd_steps)

    def get_reconstruction_cost(self, x, v_total_inputs):
        """Used to monitor progress when using CD-k"""
        p = self.v_unit.scale(v_total_inputs)
        cross_entropy = - T.sum(x * T.log(p) + (1 - x) * T.log(1 - p), axis=1)
        cost = T.mean(cross_entropy)
        return cost

    def get_pseudo_likelihood(self, x, updates):
        """Stochastic approximation to the pseudo-likelihood.
        Used to Monitor progress when using PCD-k
        Only considered the case for modelling P(X)
        but not joint distribution P(X1, X2) for association """

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

    def get_partial_derivatives(self, x, y):
        # Differentiate cost function w.r.t params to get gradients for param updates
        if self.associative:
            # Perform Gibbs Sampling to generate negative statistics
            res = self.negative_statistics(x, y)
            v_sample = res[1]
            v_input = res[2]
            v2_sample = res[5]
            v2_input = res[6]
            cost = T.mean(self.free_energy(x, y)) - T.mean(self.free_energy(v_sample, v2_sample))
            grads = T.grad(cost, self.params, consider_constant=[v_sample, v2_sample])
            stats = [v_input, v2_input]
        else:
            res = self.negative_statistics(x)
            v_sample = res[1]
            v_input = res[2]
            cost = T.mean(self.free_energy(x)) - T.mean(self.free_energy(v_sample))
            grads = T.grad(cost, self.params, consider_constant=[v_sample])

            # _, _, h = self.sample_h_given_v(x)
            # _, _, vs = self.sample_v_given_h(h)
            # _, _, hs = self.sample_h_given_v(vs)
            # dw = (T.dot(x.T, h) - T.dot(vs.T, hs)) / -self.train_parameters.batch_size
            # dv = T.mean(x - vs, axis=0)
            # dh = T.mean(h - hs, axis=0)
            # grads[1] = dv
            # grads[2] = dh
            # grads = [dw, dv, dh]
            # Differentiate cost function w.r.t params to get gradients for param updates

            stats = [v_input]

        updates = res[0]

        return {"gradients": grads,
                "updates": updates,
                "statistics": stats}

    def get_cost_updates(self, x, param_increments, y=None):
        """
        Get Cost function and a list of variables to update. To be called by get_train_fn function.
        Order of Parameters is fixed:
        1. W
        2. v1
        3. h
        4. U
        5. v2

        :param x: Theano symbolic variable for an input
        :param param_increments: Contains supplemental variables that will be used to update the variables.
                                 First 5 elements are for momentum, the last element is for sparsity constraint.

        :param y: Theano symbolic variable for an association input
        :return: cost, updates
        """

        # Cast parameters
        param = self.train_parameters
        lr = T.cast(param.learning_rate, dtype=t_float_x)
        m = T.cast(param.momentum, dtype=t_float_x)
        weight_decay = T.cast(param.weight_decay, dtype=t_float_x)

        # Declare parameter update variables
        old_ds = param_increments[:-1]

        # if self.associative:
        # old_DW, old_Dvbias, old_Dhbias, old_DU, old_Dvbias2 = old_ds
        # else:
        # old_DW, old_Dvbias, old_Dhbias = old_ds

        pre_updates = []
        if param.momentum_type is NESTEROV:
            for (p, old_dp) in zip(self.params, old_ds):
                pre_updates.append((p, p + m * old_dp))

        grad_meta = self.get_partial_derivatives(x, y)
        gradients = grad_meta["gradients"]
        updates = grad_meta["updates"]

        # For each parameters, compute: new_dx = m * old_dx - lr * grad_x
        new_ds = map(lambda (d, g): m * d - lr * g, zip(old_ds, gradients))

        # TODO
        # if not param.weight_decay_for_bias:
        # only apply weight decay to W, U

        if param.momentum_type is NESTEROV:

            def nesterov_update(p, grad_p, old_dp):
                # Nesterov update:
                # v_new = m * v_old - lr * Df(x_old + m*v_old)
                # x_new = x_old + v_new
                # <=> x_new = [x_old + m * v_old] - lr * Df([x_old + m * v_old])
                # NOTE: there is a hack here -- we compute the partial derivatives wrt. (x + m * v)
                # It means grad_x = dFd(x+ m*v) and x = (x - m * dx)
                new_p = p - lr * grad_p
                new_p = new_p - lr * weight_decay * (p - m * old_dp)
                return new_p

            new_params = map(nesterov_update, self.params, gradients, old_ds)

        else:

            def classical_update(p, new_dp):
                # Classical Momentum from Sutskever, Hinton.
                # v_new = momentum * v_old + lr * grad_wrt_w
                # w_new = w_old + v_new
                new_p = p + new_dp
                new_p -= lr * weight_decay * p
                return new_p

            new_params = map(classical_update, self.params, new_ds)

        # Sparsity
        if param.sparsity_constraint:
            active_probability_h = param_increments[-1]
            sparsity_target = T.cast(param.sparsity_target, t_float_x)
            sparsity_cost = T.cast(param.sparsity_cost, t_float_x)
            sparsity_decay_rate = T.cast(param.sparsity_decay, t_float_x)
            # 1. Compute actual probability of hidden unit being active, q
            # 1.1 Get q_current (mean probability that a unit is active in each mini-batch
            if self.associative:
                _, h_p_activation = self.prop_up(x, y)
            else:
                _, h_p_activation = self.prop_up(x)
            # q is the decaying average of mean active probability in each batch
            q = sparsity_decay_rate * active_probability_h + (1 - sparsity_decay_rate) * T.mean(h_p_activation, axis=0)

            # 1.2 Update q_current = q for next iteration
            updates[active_probability_h] = T.cast(q, t_float_x)

            # 2. Define Sparsity Penalty Measure (dim = 1 x n_hidden)
            sparsity_penalty = T.nnet.binary_crossentropy(sparsity_target, q)

            # 3. Get the derivative
            if isinstance(self.h_unit, BinaryUnit):  # if sigmoid
                d_sparsity = q - sparsity_target
            else:
                # Summation is a trick to differentiate element-wise
                # (as non relevant terms in sum vanish because they are constant w.r.t the partial derivatives)
                d_sparsity = T.grad(T.sum(sparsity_penalty), q)

            # Apply derivative scaled by sparsity_cost to the weights
            # 1. use same quantity to adjust each weight
            # new_hbias -= lr * sparsity_cost * d_sparsity
            # new_W -= lr * sparsity_cost * d_sparsity
            # 2. multiply quantity by dq/dw (chain rule)

            # chain_p = T.grad(T.sum(q), self.params, disconnected_inputs='ignore')
            # for i in xrange(len(new_params)):
            # new_params = [p - lr * sparsity_cost * d_sparsity * chain_p[i] for i, p in enumerate(new_params)]

            if self.associative:
                chain_W, chain_h, chain_U = T.grad(T.sum(q), [self.W, self.h_bias, self.U])
                new_params[0] -= T.cast(lr * sparsity_cost * d_sparsity * chain_W, t_float_x)
                new_params[2] -= lr * sparsity_cost * d_sparsity * chain_h
                new_params[3] -= lr * sparsity_cost * d_sparsity * chain_U
            else:
                chain_W, chain_h = T.grad(T.sum(q), [self.W, self.h_bias])
                new_params[0] -= T.cast(lr * sparsity_cost * d_sparsity * chain_W, t_float_x)
                new_params[2] -= T.cast(lr * sparsity_cost * d_sparsity * chain_h, t_float_x)
                #
                # chain_W, chain_h = T.grad(T.sum(q), [self.W, self.h_bias])
                # new_hbias -= lr * sparsity_cost * d_sparsity * chain_h
                # new_W -= lr * sparsity_cost * d_sparsity * chain_W

        # update parameters
        for (p, new_p) in zip(self.params, new_params):
            updates[p] = new_p

        # update velocities (used for momentum)
        for (old_dp, new_dp) in zip(old_ds, new_ds):
            updates[old_dp] = new_dp

        # Cost function
        stats = grad_meta["statistics"]
        if self.cd_type is PERSISTENT:
            # cost = self.get_reconstruction_cost(x, v_total_inputs)
            measure_cost = self.get_pseudo_likelihood(x, updates)
        else:
            v_total_inputs = stats[0]
            measure_cost = self.get_reconstruction_cost(x, v_total_inputs)
            if self.associative:
                v2_total_inputs = stats[1]
                measure_cost += self.get_reconstruction_cost(y, v2_total_inputs)

        return measure_cost, updates, pre_updates

    def get_train_fn(self, train_data, assoc_data):
        param = self.train_parameters
        batch_size = param.batch_size
        index = T.lscalar()
        x = T.matrix('x')
        y = T.matrix('y')
        param_increments = []

        # Initialise Variables used for training
        # For momentum
        old_DW = theano.shared(value=np.zeros(self.W.get_value().shape, dtype=t_float_x),
                               name='old_DW', borrow=True)
        old_Dvbias = theano.shared(value=np.zeros(self.v_n, dtype=t_float_x),
                                   name='old_Dvbias', borrow=True)
        old_Dhbias = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
                                   name='old_Dhbias', borrow=True)
        param_increments += [old_DW, old_Dvbias, old_Dhbias]

        if self.associative:
            old_DU = theano.shared(value=np.zeros(self.U.get_value().shape, dtype=t_float_x),
                                   name='old_DU', borrow=True)
            old_Dvbias2 = theano.shared(value=np.zeros(self.v_n2, dtype=t_float_x),
                                        name='old_Dvbias2', borrow=True)
            param_increments += [old_DU, old_Dvbias2]

        active_probability_h = self.active_probability_h

        # # For sparsity cost
        # active_probability_h = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
        # name="active_probability_h",
        # borrow=True)


        param_increments += [active_probability_h]

        if self.associative:
            y_val = assoc_data[index * batch_size: (index + 1) * batch_size]
        else:
            y_val = 0

        cross_entropy, updates, pre_updates = self.get_cost_updates(x, param_increments, y)
        pre_train = theano.function([], updates=pre_updates)
        train_rbm = theano.function(
            [index],
            cross_entropy,  # use cross entropy to keep track
            updates=updates,
            givens={
                x: train_data[index * batch_size: (index + 1) * batch_size],
                y: y_val
            },
            name='train_rbm',
            on_unused_input='warn'
        )

        def train_fn(i):
            pre_train()
            return train_rbm(i)

        return train_fn

    @staticmethod
    def get_sub_data(train_data, train_label, factor=10):
        l = train_data.get_value(borrow=True).shape[0]
        nl = max(1, int(l / factor))
        sub_data = theano.shared(train_data.get_value(borrow=True)[0: nl])
        sub_label = None
        if train_label:
            sub_label = theano.shared(train_label.get_value(borrow=True)[0: nl])

        return sub_data, sub_label

    def pretrain(self, x, y=None):
        self.pretrain_lr(x, y)
        self.pretrain_mean_activity_h(x, y)

    def set_hidden_mean_activity(self, x, y=None):
        print '... Sparsity: setting initial mean activity for hidden units'
        sub_x, sub_y = self.get_sub_data(x, y)
        _, ph = self.prop_up(sub_x, sub_y)
        active_probability_h = theano.function([], T.mean(ph, axis=0))().astype(t_float_x)
        print active_probability_h.shape
        self.active_probability_h = theano.shared(active_probability_h, 'active_probability_h')
        # print active_probability_h

    def pretrain_mean_activity_h(self, x, y=None):
        print '... adjusting mean activity'
        sub_x, sub_y = self.get_sub_data(x, y)
        self.train_parameters.sparsity_constraint = True
        self.train_parameters.sparsity_cost = 0.01

        for i in xrange(10):
            print 'attempt {}'.format(i)
            self.train(x, y)

            _, ph = self.prop_up(sub_x, sub_y)
            mean_ph = T.mean(ph, axis=0)
            f = theano.function([], mean_ph)
            active_probability_h = f()
            self.active_probability_h = theano.shared(active_probability_h, 'active_probability_h')

            print 'mean, max, min', np.mean(active_probability_h), np.max(active_probability_h), np.min(
                active_probability_h)

            self.set_default_weights()
            if np.mean(np.abs(active_probability_h - self.train_parameters.sparsity_target)) < 0.1:
                print 'Sparsity cost set to: {}'.format(self.train_parameters.sparsity_cost)
                break

            self.train_parameters.sparsity_cost *= 10
            print 'sparsity cost updated to {}'.format(self.train_parameters.sparsity_cost)

    def set_default_weights(self):
        self.W = self.get_initial_weight(None, self.v_n, self.h_n, 'W')
        self.v_bias = self.get_initial_bias(None, self.v_n, 'v_bias')
        self.h_bias = self.get_initial_bias(None, self.h_n, 'h_bias')
        self.params = [self.W, self.v_bias, self.h_bias]
        if self.associative:
            self.U = self.get_initial_weight(None, self.v_n, self.h_n, 'U')
            self.v_bias2 = self.get_initial_bias(None, self.v_n2, 'v_bias2')
            self.params = self.params + [self.U, self.v_bias2]

    def pretrain_lr(self, x, y=None):
        '''
        From Hinton -- learning rate should be weights * 10^-3
        :param x:
        :param y:
        :return:
        '''
        # train with subdata
        sub_data, sub_label = self.get_sub_data(x, y)

        # Retrieve parameters
        tr = self.train_parameters
        self.train_parameters.learning_rate = 0.1

        # Train
        self.train(sub_data, sub_label)

        # Analyse
        avg_hist, avg_bins = np.histogram(np.abs(self.track_progress.weight_hist['avg']),
                                          bins=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
        adjusted_lr = np.mean(np.abs(self.track_progress.weight_hist['avg'])) * tr.adj_lr
        print "new learning rate: {}".format(adjusted_lr)

        # update the learning rate
        tr.learning_rate = adjusted_lr
        self.train_parameters = tr
        self.set_default_weights()

    def train(self, train_data, train_label=None):
        self.training = True
        """Trains RBM. For now, input needs to be Theano matrix"""
        param = self.train_parameters
        batch_size = param.batch_size
        mini_batches = train_data.get_value(borrow=True).shape[0] / batch_size
        train_fn = self.get_train_fn(train_data, train_label)

        plotting_time = 0.
        start_time = time.clock()  # Measure training time
        for epoch in xrange(param.epochs):
            mean_cost = []
            for batch_index in xrange(mini_batches):
                if self.dropout:
                    # self.dropout_mask = self.np_rand.binomial(n=1, p=self.dropout_rate, size=self.h_n).astype(t_float_x)
                    self.dropout_mask = self.np_rand.binomial(n=1, p=self.dropout_rate, size=(batch_size,self.h_n)).astype(t_float_x)

                cost = train_fn(batch_index)
                if not math.isnan(cost):
                    # continue
                    # raise Exception('training cost is infty -- try lowering learning rate')

                    mean_cost += [cost]
                if self.track_progress and self.track_progress.monitor_weights:
                    self.track_progress.monitor_wt(self)
                    self.track_progress.monitor_mean_activity(self, train_data, train_label)

            if self.track_progress:
                print 'Epoch %d, cost is ' % epoch, np.mean(mean_cost)
                plotting_time += self.track_progress.visualise_weight(self, 'epoch_%i.png' % epoch)

        end_time = time.clock()
        pre_training_time = (end_time - start_time) - plotting_time

        if self.track_progress:
            print ('Training took %f minutes' % (pre_training_time / 60.))
            if self.track_progress.monitor_weights:
                print 'Weight histogram'
                # avg_hist, avg_bins = np.histogram(np.abs(self.track_progress.weight_hist['avg']), bin=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
                avg_hist, avg_bins = np.histogram(self.track_progress.weight_hist['avg'])
                std_hist, std_bins = np.histogram(self.track_progress.weight_hist['std'])
                print avg_hist, avg_bins
                print std_hist, avg_bins
                print self.track_progress.weight_hist['min']
                print self.track_progress.weight_hist['max']

        self.training = False
        return [mean_cost]

    def save(self):
        store.store_object(self)
        print "... saved RBM object to " + os.getcwd() + "/" + str(self)

    def sample(self, n=1, k=1, p=0.01, rand_type='uniform'):
        assert rand_type in ['binomial', 'uniform', 'normal', 'mean','noisy_mean']

        # Generate random "v"
        if rand_type == 'binomial':
            data = self.np_rand.binomial(size=(n, self.v_n), n=1, p=p).astype(t_float_x)
        elif rand_type == 'mean':
            data = np.tile(p, (n, 1))
        elif rand_type == 'noisy_mean':
            data = np.tile(p, (n, 1)) + self.np_rand.normal(0, 0.2, size=(n, self.v_n)).astype(t_float_x)
        else: # rand_type == 'uniform':
            data = self.np_rand.uniform(size=(n, self.v_n), low=0, high=1).astype(t_float_x)

        return self.reconstruct(data, k)

    def reconstruct(self, data, k=1, plot_n=None, plot_every=1, img_name='reconstruction'):
        '''
        Reconstruct image given cd-k
        - data: theano
        '''
        if utils.isSharedType(data):
            orig = data.get_value(borrow=True)
        else:
            orig = data

        # Set the initial chain
        chain_state = theano.shared(np.asarray(orig, dtype=theano.config.floatX), name='reconstruct_root')

        # Gibbs sampling
        k_batch = k / plot_every
        (res, updates) = theano.scan(self.gibbs_vhv,
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
            reconstructions.append(reconstruction_chain[-1])

        if self.track_progress:
            self.track_progress.visualise_reconstructions(orig, reconstructions, plot_n, img_name=img_name)

        return reconstructions[-1]

    def reconstruct_association(self, x, y=None, k=1, bit_p=0, plot_n=None, plot_every=1,
                                img_name='association_reconstruction.png'):
        # Initialise parameters
        if not utils.isSharedType(x):
            x = theano.shared(x, allow_downcast=True)
        data_size = x.get_value().shape[0]
        if not y:
            y = self.rand.binomial(size=(data_size, self.v_n2), n=1, p=bit_p, dtype=t_float_x)

        # Gibbs sampling
        _, _, h_sample = self.sample_h_given_v(x, y)
        chain_start = theano.shared(theano.function([], h_sample)())
        k_batch = k / plot_every
        (res, updates) = theano.scan(
            self.gibbs_hvh_fixed,
            outputs_info=[None, None, None, None, None,
                          None, None, None, chain_start],
            non_sequences=[x], n_steps=plot_every, name="Gibbs_sampling_association"
        )
        updates.update({chain_start: res[-1][-1]})
        gibbs_sampling_assoc = theano.function([], res, updates=updates)

        # Runner
        reconstructions = []
        for i in xrange(k_batch):
            result = gibbs_sampling_assoc()
            [_, _, _, _, reconstruction_chain, _, _, _, _] = result
            reconstructions.append(reconstruction_chain[-1])

        if self.track_progress:
            self.track_progress.visualise_reconstructions(x.get_value(borrow=True), reconstructions, plot_n,
                                                          img_name=img_name)

        return reconstruction_chain[-1]

    def reconstruct_association_opt(self, x, y=None, k=1, bit_p=0, plot_n=None, plot_every=1,
                                    img_name='association_reconstruction.png'):
        '''
        As an optimisation, we can concatenate two images and feed it as a single image to train the network.
        In this way theano performs matrix optimisation so its much faster.

        If such optimisation was done, this reconstruction method should be used.
        '''

        # Initialise parameters
        if not utils.isSharedType(x):
            x = theano.shared(x, allow_downcast=True)
        data_size = x.get_value().shape[0]
        if not y:
            y = self.rand.binomial(size=(data_size, self.v_n / 2), n=1, p=bit_p, dtype=t_float_x)

        # Concatenate x and y
        z = T.concatenate([x, y], axis=1)

        # Gibbs sampling
        _, _, h_sample = self.sample_h_given_v(z)
        chain_start = theano.shared(theano.function([], h_sample)(), name='Z')

        print chain_start.get_value().shape

        k_batch = k / plot_every

        (res, updates) = theano.scan(
            self.gibbs_hvh_fixed2,
            outputs_info=[None, None, None, None, None, chain_start],
            non_sequences=[x], n_steps=plot_every, name="Gibbs_sampling_association"
        )
        updates.update({chain_start: res[-1][-1]})
        gibbs_sampling_assoc = theano.function([], res, updates=updates)

        # Runner
        reconstructions = []
        for i in xrange(k_batch):
            result = gibbs_sampling_assoc()
            [_, reconstruction_chain, _, _, _, _] = result
            reconstructions.append(reconstruction_chain[-1][:, (self.v_n / 2):])

        if self.track_progress:
            self.track_progress.visualise_reconstructions(x.get_value(borrow=True), reconstructions, plot_n,
                                                          img_name=img_name, opt=True)

        return reconstruction_chain[-1][:, (self.v_n / 2):]

    def mean_field_inference(self, x, tolerance=0.01, sample=False, k=100, bit_p=0, plot_n=None, plot_every=1,
                             img_name='mean_field'):
        # Initialise parameters
        if not utils.isSharedType(x):
            x = theano.shared(x, allow_downcast=True)
        data_size = x.get_value().shape[0]

        plot_every = 100
        mu = theano.shared(np.zeros((data_size, self.v_n2), dtype=t_float_x), name='mu')
        tau = theano.shared(np.zeros((data_size, self.h_n), dtype=t_float_x), name='tau')

        # Mean field inference -- basically gibbs sampling with no actual sampling
        def mean_field(m, t, x):
            _, mu2 = self.prop_down_assoc(t)
            _, tau2 = self.prop_up(x, mu2)
            return mu2, tau2  # , {ctr: ctr+1}, theano.scan_module.until(ctr < 50)

        def mean_field_rev(m, t, x):
            _, tau2 = self.prop_up(x, m)
            _, mu2 = self.prop_down_assoc(tau2)
            return mu2, tau2  # , {ctr: ctr+1}, theano.scan_module.until(ctr < 50)

        k_batch = k / plot_every
        (res, updates) = theano.scan(
            mean_field_rev,
            outputs_info=[mu, tau],
            non_sequences=[x], n_steps=plot_every, name="mean_field_inference"
        )
        updates.update({tau: res[1][-1]})
        updates.update({mu: res[0][-1]})
        mfi = theano.function([], res, updates=updates)

        # Runner
        reconstructions = []
        for i in xrange(k_batch):
            m, t = mfi()
            reconstructions.append(m[-1])

        if self.track_progress:
            self.track_progress.visualise_reconstructions(x.get_value(borrow=True), reconstructions, plot_n,
                                                          img_name=img_name)

        if sample:
            return self.np_rand.binomial(n=1, p=m[-1]).astype(t_float_x)
        else:
            return m[-1]

    def mean_field_inference_opt(self, x, y=None, ylen=-1, sample=False, k=100, img_name='mean_field_inference'):
        '''
        As an optimisation, we can concatenate two images and feed it as a single image to train the network.
        In this way theano performs matrix optimisation so its much faster.

        If such optimisation was done, this reconstruction method should be used.
        '''

        plot_n = 100
        plot_every = k if k <= 10 else 10
        ylen = self.v_n / 2 if type(y) is None and ylen == -1 else self.v_n - len(y.get_value()[0]) if type(y) is not None else ylen

        # Initialise parameters
        if not utils.isSharedType(x):
            x = theano.shared(x, allow_downcast=True)
        data_size = x.get_value().shape[0]
        if type(y) is None:
            y = self.rand.binomial(size=(data_size, ylen), n=1, p=0, dtype=t_float_x)

        # get initial values of tau (Concatenate x and y)
        z = T.concatenate([x, y], axis=1)
        _, tau = self.prop_up(z)
        chain_start = theano.shared(theano.function([], tau)(), name='tau')

        # mean field func
        def mean_field(tau1, fixed):
            _, mu2 = self.prop_down(tau1)
            mu2 = T.concatenate([fixed, mu2[:, (ylen):]], axis=1)
            _, tau2 = self.prop_up(mu2)
            return mu2, tau2  # , {ctr: ctr+1}, theano.scan_module.until(ctr < 50)

        # loop
        k_batch = k / plot_every
        (res, updates) = theano.scan(
            mean_field,
            outputs_info=[None, chain_start],
            non_sequences=[x], n_steps=plot_every, name="mean_field_opt"
        )
        updates.update({chain_start: res[-1][-1]})
        mean_field_opt = theano.function([], res, updates=updates)

        # Runner
        reconstructions = []
        for i in xrange(k_batch):
            result = mean_field_opt()
            [reconstruction_chain, _] = result
            reconstructions.append(reconstruction_chain[-1][:, (ylen):])

        if self.track_progress:
            self.track_progress.visualise_reconstructions(x.get_value(borrow=True), reconstructions, plot_n,
                                                          img_name=img_name, opt=True)

        if sample and type(self.v_unit) is RBMUnit:
            return self.np_rand.binomial(n=1, p=reconstruction_chain[-1][:, (ylen):])
        else:
            return reconstruction_chain[-1][:, (ylen):]

    def classify(self, xs):
        assert self.associative
        n_case = xs.get_value().shape[0]
        n_classes = self.v_n2
        pred_mat = np.zeros((n_case, n_classes))
        for c in xrange(n_classes):
            v_c = utils.get_class_vector(c, n_classes)
            vs = np.tile(v_c, (n_case, 1))
            energy = theano.function([], self.free_energy(xs, vs))()
            print energy
            pred_mat[:, c] = energy

        print np.sum(pred_mat)
        print pred_mat / np.sum(pred_mat, axis=0)
        return pred_mat.argmin(axis=0)




class AssociativeRBM(RBM):
    pass


def test_rbm():
    print "Testing RBM"

    # Load mnist hand digits
    datasets = m_loader.load_digits(n=[100, 0, 100], digits=[1])
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initialise the RBM and training parameters
    tr = rbm_config.TrainParam(learning_rate=0.01,
                               momentum_type=NESTEROV,
                               momentum=0.5,
                               weight_decay=0.01,
                               sparsity_constraint=True,
                               sparsity_target=0.01,
                               sparsity_cost=0.01,
                               sparsity_decay=0.1)

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 2

    rbm = RBM(n_visible,
              n_visible,
              n_hidden,
              associative=False,
              cd_type=CLASSICAL,
              cd_steps=1,
              v_activation_fn=rectify,
              h_activation_fn=rectify,
              train_parameters=tr,
              progress_logger=ProgressLogger())

    print "... initialised RBM"

    curr_dir = store.move_to(str(rbm))
    print "... moved to {}".format(curr_dir)

    # Train RBM
    rbm.train(train_set_x)

    # Test RBM
    rbm.reconstruct(test_set_x, k=1, n=20)


if __name__ == '__main__':
    test_rbm()
