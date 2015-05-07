import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from theano.tensor.shared_randomstreams import RandomStreams
from utils import *

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
# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'


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
                 v_n=10,
                 v_n2=10,
                 h_n=10,
                 associative=False,
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

        self.np_rand = np.random.RandomState(123)
        self.rand = RandomStreams(self.np_rand.randint(2 ** 30))

        W = self.get_initial_weight(W, v_n, h_n, 'W')
        v_bias = self.get_initial_bias(v_bias, v_n, 'v_bias')
        h_bias = self.get_initial_bias(h_bias, h_n, 'h_bias')

        if train_parameters is None:
            train_parameters = TrainParam()

        persistent = theano.shared(value=np.zeros((train_parameters.batch_size, h_n),
                                                  dtype=t_float_x), name="persistent")

        self.train_parameters = train_parameters

        # Weights
        self.W = W
        # Visible Layer
        self.v_n = v_n
        self.v_bias = v_bias
        self.v_activation_fn = v_activation_fn
        # Hidden Layer
        self.h_n = h_n
        self.h_bias = h_bias
        self.h_activation_fn = h_activation_fn
        # Gibbs Sampling Method
        self.cd_type = cd_type
        self.cd_steps = cd_steps
        self.persistent = persistent
        self.params = [self.W, self.v_bias, self.h_bias]

        self.associative = associative
        if associative:
            self.U = self.get_initial_weight(U, v_n2, h_n, 'U')
            # Visible Layer 2
            self.v_n2 = v_n2
            self.v_bias2 = self.get_initial_bias(v_bias2, v_n2, 'v_bias2')
            self.v_activation_fn2 = v_activation_fn2
            self.params += [self.U, self.v_bias2]

    def get_initial_weight(self, w, nrow, ncol, name):
        if w is None:
            w = np.asarray(
                self.np_rand.uniform(
                    low=-4 * np.sqrt(6. / (nrow + ncol)),
                    high=4 * np.sqrt(6. / (nrow + ncol)),
                    size=(nrow, ncol)
                ),
                dtype=t_float_x
            )
        if 'numpy' in str(type(w)):
            w = theano.shared(value=w, name=name, borrow=True)
        return w

    def get_initial_bias(self, bias, n, name):
        if bias is None:
            bias = np.zeros(n, dtype=t_float_x)

        if 'numpy' in str(type(bias)):
            bias = theano.shared(value=bias, name=name, borrow=True)
        return bias

    def __str__(self):
        return "rbm_" + str(self.h_n) + \
               "_" + self.cd_type + str(self.cd_steps) + \
               "_" + str(self.train_parameters)

    def free_energy(self, v, v2=None):
        if self.associative:
            return self.calc_free_energy(v, v2)
        else:
            return self.calc_free_energy(v)

    def calc_free_energy(self, v, v2=None):
        w = self.W
        v_bias = self.v_bias
        h_bias = self.h_bias

        t0 = - T.dot(v, v_bias)
        t1 = T.dot(v, w) + h_bias

        if v2:
            u = self.U
            v_bias2 = self.v_bias2
            t0 += -T.dot(v2, v_bias2)
            t1 += T.dot(v2, u)

        t2 = - T.sum(T.log(1 + T.exp(t1)))

        return t0 + t2

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

    # def gibbs_sampling(self, chain_start, sample_fn, k):
    #     return theano.scan(
    #         self.sample_fn,
    #         outputs_info=[None, None, None, None, None, chain_start],
    #         n_steps=k
    #     )

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
            if y:
                return self.pcd_assoc(self.cd_steps)
            else:
                return self.pcd(self.cd_steps)
        elif y:
            return self.contrastive_divergence_assoc(x, y, 1)
        else:
            return self.contrastive_divergence(x, self.cd_steps)

    def get_reconstruction_cost(self, x, v_total_inputs):
        """Used to monitor progress when using CD-k"""
        p = self.v_activation_fn(v_total_inputs)
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
            # Differentiate cost function w.r.t params to get gradients for param updates
            cost = T.mean(self.free_energy(x)) - T.mean(self.free_energy(v_sample))
            grads = T.grad(cost, self.params, consider_constant=[v_sample])
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
        #     old_DW, old_Dvbias, old_Dhbias, old_DU, old_Dvbias2 = old_ds
        # else:
        #     old_DW, old_Dvbias, old_Dhbias = old_ds

        pre_updates = []
        if param.momentum_type is NESTEROV:
            for (p, old_dp) in zip(self.params, old_ds):
                pre_updates.append((p, p + m * old_dp))

        grad_meta = self.get_partial_derivatives(x, y)
        gradients = grad_meta["gradients"]
        updates = grad_meta["updates"]

        # For each parameters, compute: new_dx = m * old_dx - lr * grad_x
        new_ds = map(lambda (d, g): m * d - lr * g, zip(old_ds, gradients))

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
            sparsity_target = param.sparsity_target
            sparsity_cost = param.sparsity_cost
            sparsity_decay_rate = param.sparsity_decay
            # 1. Compute actual probability of hidden unit being active, q
            # 1.1 Get q_current (mean probability that a unit is active in each mini-batch
            if self.associative:
                _, h_p_activation = self.prop_up(x, y)
            else:
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

            # Apply derivative scaled by sparsity_cost to the weights
            # 1. use same quantity to adjust each weight
            # new_hbias -= lr * sparsity_cost * d_sparsity
            # new_W -= lr * sparsity_cost * d_sparsity
            # 2. multiply quantity by dq/dw (chain rule)

            chain_p = T.grad(T.sum(q), self.params, disconnected_inputs='ignore')
            for i in xrange(len(new_params)):
                new_params = [p - lr * sparsity_cost * d_sparsity * chain_p[i] for i, p in enumerate(new_params)]

            # if self.associative:
            #     chain_W, chain_h, chain_U = T.grad(T.sum(q), [self.W, self.h_bias, self.U])
            #     new_params[0] -= lr * sparsity_cost * d_sparsity * chain_W
            #     new_params[2] -= lr * sparsity_cost * d_sparsity * chain_h
            #     new_params[3] -= lr * sparsity_cost * d_sparsity * chain_U
            # else:
            #     chain_W, chain_h, chain_U = T.grad(T.sum(q), [self.W, self.h_bias])
            #     new_params[0] -= lr * sparsity_cost * d_sparsity * chain_W
            #     new_params[2] -= lr * sparsity_cost * d_sparsity * chain_h

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

        # For sparsity cost
        active_probability_h = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
                                             name="active_probability_h",
                                             borrow=True)


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

    def train(self, train_data, train_label=None):
        """Trains RBM. For now, input needs to be Theano matrix"""

        param = self.train_parameters
        batch_size = param.batch_size
        mini_batches = train_data.get_value(borrow=True).shape[0] / batch_size

        if self.associative:
            train_fn = self.get_train_fn(train_data, train_label)
        else:
            train_fn = self.get_train_fn(train_data, None)

        plotting_time = 0.
        start_time = time.clock()       # Measure training time
        for epoch in xrange(param.epochs):
            mean_cost = []
            for batch_index in xrange(mini_batches):
                mean_cost += [train_fn(batch_index)]

            print 'Epoch %d, cost is ' % epoch, np.mean(mean_cost)

            if param.plot_during_training and self.v_n == 784:
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

    def plot_samples(self, data, image_name='samples.png', plot_every=1000):
        n_chains = 20   # Number of Chains to perform Gibbs Sampling
        n_samples_from_chain = 10  # Number of samples to take from each chain

        test_set_size = data.get_value(borrow=True).shape[0]
        rand = np.random.RandomState(1234)
        test_input_index = rand.randint(test_set_size - n_chains)
        # Sample after 1000 steps of Gibbs Sampling each time
        persistent_vis_chain = theano.shared(
            np.asarray(
                data.get_value(borrow=True)[test_input_index:test_input_index + n_chains],
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
        image.save(image_name)

    def save(self):
        datastorage.store_object(self)
        print "... saved RBM object to " + os.getcwd() + "/" + str(self)

    def sample(self, n=1, k=1):
        # Generate random "v"
        activation_probability = 0.1
        data = self.rand.binomial(size=(n, self.v_n), n=1,
                                  p=activation_probability, dtype=t_float_x).eval()
        return self.reconstruct(data, k)

    def reconstruct(self, data, k=1, image_name="reconstructions.png"):
        # data_size = data.get_value(borrow=True).shape[0]
        data_size = data.shape[0]
        if self.associative:
            return self.reconstruct_association(data, None, k)

        # Gibbs sampling
        x = T.matrix("x")
        chain_start = x
        (res, updates) = theano.scan(self.gibbs_vhv,
                                     outputs_info=[None, None, None,
                                                   None, None, chain_start],
                                     n_steps=k,
                                     name="Gibbs_sampling_reconstruction")

        gibbs_sampling = theano.function([x], res, updates=updates)
        result = gibbs_sampling(data)

        [h_total_input,
         h_p_activation,
         h_sample,
         v_total_input,
         v_p_activation,
         v_sample] = result

        if self.v_n == 784:
            image_data = np.zeros(
                (29 * (k+1) + 1, 29 * data_size - 1),
                dtype='uint8'
            )

            # Original images
            image_data[0:28, :] = tile_raster_images(
                X=data,
                img_shape=(28, 28),
                tile_shape=(1, data_size),
                tile_spacing=(1, 1)
            )

            # Generate image by plotting the sample from the chain
            for i in xrange(1, k):
                vis_mf = v_p_activation[i]
                print ' ... plotting sample ', i
                image_data[29 * i:29 * i + 28, :] = tile_raster_images(
                    X=vis_mf,
                    img_shape=(28, 28),
                    tile_shape=(1, data_size),
                    tile_spacing=(1, 1)
                )

            # construct image
            image = Image.fromarray(image_data)
            image.save(image_name)

            for i in xrange(k):
                vis_mf = v_sample[i]
                print ' ... plotting sample ', i
                image_data[29 * i:29 * i + 28, :] = tile_raster_images(
                    X=vis_mf,
                    img_shape=(28, 28),
                    tile_shape=(1, data_size),
                    tile_spacing=(1, 1)
                )

            image = Image.fromarray(image_data)
            image.save("reconstructed_sample.png")
        return v_sample[-1]

    def reconstruct_association(self, x, y=None, k=1, bit_p=0, image_name="association.png", sample_size=None):
        data_size = x.get_value().shape[0]
        if not sample_size:
            sample_size = data_size
        data = T.matrix("data_x")
        association = T.matrix("association")

        if not y:
            y = self.rand.binomial(size=(data_size, self.v_n2), n=1, p=bit_p, dtype=t_float_x)

        h_total_input, h_p_activation, h_sample = self.sample_h_given_v(data, association)
        chain_start = h_sample
        (
            res,
            updates
        ) = theano.scan(
            self.gibbs_hvh_fixed,
            outputs_info=[None, None, None, None, None,
                          None, None, None, chain_start],
            non_sequences=[data],
            n_steps=k,
            name="Gibbs_sampling_association"
        )

        gibbs_sampling_assoc = theano.function([], res,
                                               updates=updates,
                                               givens={
                                                   data: x,
                                                   association: y
                                               })

        result = gibbs_sampling_assoc()

        [v_total_inputs,
         v_p_activations,
         v_samples,
         v2_total_inputs,
         v2_p_activations,
         v2_samples,
         h_total_inputs,
         h_p_activations,
         h_samples] = result

        if self.v_n == 784:
            image_data = np.zeros(
                (29 * (k+1) + 1, 29 * sample_size - 1),
                dtype='uint8'
            )

            # Original images
            image_data[0:28, :] = tile_raster_images(
                X=x.get_value(borrow=True)[0:sample_size],
                img_shape=(28, 28),
                tile_shape=(1, sample_size),
                tile_spacing=(1, 1)
            )

            # Generate image by plotting the sample from the chain
            for i in xrange(1, k):
                # vis_mf = v2_samples[i]
                vis_mf = v2_p_activations[i]
                print ' ... plotting sample ', i
                image_data[29 * i:29 * i + 28, :] = tile_raster_images(
                    X=vis_mf[0:sample_size],
                    img_shape=(28, 28),
                    tile_shape=(1, sample_size),
                    tile_spacing=(1, 1)
                )

            # construct image
            image = Image.fromarray(image_data)
            image.save(image_name)

        return v2_p_activations[-1]


def test_rbm():
    print "Testing RBM"

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.01,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.01,
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_cost=0.01,
                    sparsity_decay=0.1,
                    plot_during_training=True)

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 15

    rbm = RBM(n_visible,
              n_visible,
              n_hidden,
              associative=False,
              cd_type=PERSISTENT,
              cd_steps=1,
              train_parameters=tr)

    print "... initialised RBM"

    rbm.move_to_output_dir()

    # Train RBM
    rbm.train(train_set_x)

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

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.001,
                    momentum_type=CLASSICAL,
                    momentum=0.1,
                    weight_decay=0.0005,
                    plot_during_training=True,
                    output_directory="AssociationLabelTest",
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_cost=0.5,
                    sparsity_decay=0.9,
                    epochs=15)

    n_visible = train_set_x.get_value().shape[1]
    n_visible2 = 10
    n_hidden = 500

    rbm = RBM(n_visible,
               n_visible2,
              n_hidden,
              associative=True,
              cd_type=PERSISTENT,
              cd_steps=1,
              train_parameters=tr)

    rbm.move_to_output_dir()

    loaded = datastorage.retrieve_object(str(rbm))
    if loaded:
        rbm = loaded
        print "... loaded precomputed rbm"
    else:
        rbm.train(train_set_x, new_train_set_y)
        rbm.save()

    rbm.associative = False
    rbm.reconstruct(test_set_x.get_value(borrow=True)[0:100], 100)
    rbm.associative = True

    # Classification test - Reconstruct y through x
    x_in = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True),
            dtype=t_float_x
        )
    )

    output_probability = rbm.reconstruct_association(x_in, None)
    sol = test_set_y.eval()
    guess = [np.argmax(lab) for lab in output_probability]
    diff = np.count_nonzero(sol - guess)

    print diff
    print diff / float(test_set_y.eval().shape[0])


def test_rbm_association():
    print "Testing Associative RBM which tries to learn even-oddness of numbers"

    # Even odd test
    testcases = 1000
    k = 1

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.04,
                    momentum_type=CLASSICAL,
                    momentum=0.01,
                    weight_decay=0.001,
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
              associative=True,
              cd_type=CLASSICAL,
              cd_steps=1,
              train_parameters=tr)

    rbm.move_to_output_dir()

    # Find 1 example which train_set_x[i] represents 0 and 1
    zero_idx = np.where(train_set_y.eval() == 0)[0]
    one_idx = np.where(train_set_y.eval() == 1)[0]
    zero_image = train_set_x.get_value(borrow=True)[zero_idx[0]]
    one_image = train_set_x.get_value(borrow=True)[one_idx[0]]

    save_digit(zero_image, "zero.png")
    save_digit(one_image, "one.png")


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
    loaded = datastorage.retrieve_object(str(rbm))
    if not loaded:
        # Train RBM - learn joint distribution
        rbm.train(train_set_x, new_train_set_y)
        rbm.save()
    else:
        rbm = loaded
        print "... loaded"

    # Reconstruct y through x
    x_in = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[0:testcases],
            # test_set_x.get_value(borrow=True),
            dtype=t_float_x
        )
    )

    print "... reconstruction of associated images"
    reconstructed_y = rbm.reconstruct_association(x_in, None, k, 0.1, sample_size=100)
    print "... reconstructed"

    # Create Dataset to feed into logistic regression

    # Train data: get only 0's and 1's
    ty = train_set_y.eval()
    zero_ones = (ty == 0) | (ty == 1)  # Get indices which the label is 0 or 1
    train_x = theano.shared(train_set_x.get_value(True)[zero_ones])
    train_y = theano.shared(ty[zero_ones])

    # Validation setL get only 0's and 1's
    ty = valid_set_y.eval()
    zero_ones = (ty == 0) | (ty == 1)
    valid_x = theano.shared(valid_set_x.get_value(True)[zero_ones])
    valid_y = theano.shared(ty[zero_ones])

    # Test set: reconstructed y's become the input. Get the corresponding x's and y's

    test_x = theano.shared(reconstructed_y[0:testcases])
    test_y = theano.shared(np.array(map(lambda x: x % 2, train_set_y.eval()), dtype=np.int32)[0:testcases])

    dataset = ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))

    # Classify the reconstructions
    logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset, 600)

    # Move back to root
    os.chdir(root_dir)
    print "moved to ... " + root_dir


if __name__ == '__main__':
    test_rbm_association()
    # test_rbm()
    # test_rbm_association_with_label()
