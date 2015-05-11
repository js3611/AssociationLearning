import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from rbm_units import *
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import mnist_loader as loader
import datastorage as store

import os
import time
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
                 learning_rate=0.1,
                 momentum_type=NESTEROV,
                 momentum=0.5,
                 weight_decay=0.001,
                 sparsity_constraint=True,
                 sparsity_target=0.01,      # in range (0.1^9, 0.01)
                 sparsity_cost=0.01,
                 sparsity_decay=0.1,
                 output_directory=None
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
        # Meta
        self.output_directory = output_directory

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

def visualise_reconstructions(orig, reconstructions, img_shape, plot_n=None, img_name='reconstruction'):
            k = len(reconstructions)
            assert k > 0

            data_size = plot_n if plot_n else orig.shape[0]

            if img_shape == (28, 28):

                image_data = np.zeros(
                    (29 * (k+1) + 1, 29 * data_size - 1),
                    dtype='uint8'
                )

                # Original images
                image_data[0:28, :] = utils.tile_raster_images(
                    X=orig,
                    img_shape=(28, 28),
                    tile_shape=(1, data_size),
                    tile_spacing=(1, 1)
                )

                # Generate image by plotting the sample from the chain
                for i in xrange(1, k+1):
                    print ' ... plotting sample ', i
                    image_data[29 * i:29 * i + 28, :] = utils.tile_raster_images(
                        X=(reconstructions[i-1]),
                        img_shape=(28, 28),
                        tile_shape=(1, data_size),
                        tile_spacing=(1, 1)
                    )

                # construct image
                image = Image.fromarray(image_data)
                image.save(img_name + '_{}.png'.format(k))


class ProgressLogger(object):

        def __init__(self,
                     likelihood=True,
                     time_training=True,
                     plot=True,
                     plot_info=None,
                     out_dir=None
                     ):

            self.likelihood = likelihood
            self.time_training = time_training
            self.plot = plot
            self.plot_info = plot_info
            self.out_dir = out_dir
            self.img_shape = (28, 28)

        def visualise_weight(self, rbm, image_name):
            plotting_start = time.clock()  # Measure plotting time

            if rbm.v_n == 784:
                tile_shape = (rbm.h_n / 10 + 1, 10)

                image = Image.fromarray(
                    utils.tile_raster_images(
                        X=rbm.W.get_value(borrow=True).T,
                        img_shape=(28, 28),
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                )
                image.save(image_name)

            plotting_end = time.clock()
            return plotting_end - plotting_start

        def visualise_reconstructions(self, orig, reconstructions, plot_n=None, img_name='reconstruction'):
            visualise_reconstructions(orig, reconstructions, (28, 28), plot_n, img_name)


class AssociationProgressLogger(ProgressLogger):
    def visualise_weight(self, rbm, image_name):
            assert rbm.associative
            if rbm.v_n == 784:
                plotting_start = time.clock()  # Measure plotting time

                w = rbm.W.get_value(borrow=True).T
                u = rbm.U.get_value(borrow=True).T

                weight = np.hstack((w, u))

                tile_shape = (rbm.h_n / 10 + 1, 10)

                image = Image.fromarray(
                    utils.tile_raster_images(
                        X=weight,
                        img_shape=(28 * 2, 28),
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                )
                image.save(image_name)

                plotting_end = time.clock()
                return plotting_end - plotting_start
            return 0

    def visualise_reconstructions(self, orig, reconstructions, plot_n=None, img_name='association'):
        visualise_reconstructions(orig, reconstructions, (28, 28), plot_n, img_name)


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
                 train_parameters=None,
                 progress_logger=None):

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

        self.track_progress = progress_logger

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

    def __str__(self):
        name = 'ass_' if self.associative else ''
        return name + "rbm_" + str(self.h_n) + \
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

    def train(self, train_data, train_label=None):
        """Trains RBM. For now, input needs to be Theano matrix"""

        param = self.train_parameters
        batch_size = param.batch_size
        mini_batches = train_data.get_value(borrow=True).shape[0] / batch_size
        train_fn = self.get_train_fn(train_data, train_label)

        plotting_time = 0.
        start_time = time.clock()       # Measure training time
        for epoch in xrange(param.epochs):
            mean_cost = []
            for batch_index in xrange(mini_batches):
                mean_cost += [train_fn(batch_index)]

            if self.track_progress:
                print 'Epoch %d, cost is ' % epoch, np.mean(mean_cost)
                plotting_time += self.track_progress.visualise_weight(self, 'epoch_%i.png' % epoch)

        end_time = time.clock()
        pre_training_time = (end_time - start_time) - plotting_time

        if self.track_progress:
            print ('Training took %f minutes' % (pre_training_time / 60.))

        return [mean_cost]

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
            image_data[29 * i:29 * i + 28, :] = utils.tile_raster_images(
                X=vis_mf,
                img_shape=(28, 28),
                tile_shape=(1, n_chains),
                tile_spacing=(1, 1)
            )

        # construct image
        image = Image.fromarray(image_data)
        image.save(image_name)

    def save(self):
        store.store_object(self)
        print "... saved RBM object to " + os.getcwd() + "/" + str(self)

    def sample(self, n=1, k=1):
        # Generate random "v"
        activation_probability = 0.1
        data = self.rand.binomial(size=(n, self.v_n), n=1,
                                  p=activation_probability, dtype=t_float_x).eval()
        return self.reconstruct(data, k)

    def reconstruct(self, data, k=1, plot_n=None, plot_every=1):
        '''
        Reconstruct image given cd-k
        - data: theano
        '''
        if 'Tensor' not in str(type(data)):
            data = theano.shared(data, allow_downcast=True)
        orig = data.get_value(borrow=False)

         # Gibbs sampling
        k_batch = k / plot_every
        (res, updates) = theano.scan(self.gibbs_vhv,
                                     outputs_info=[None, None, None,
                                                   None, None, data],
                                     n_steps=plot_every,
                                     name="Gibbs_sampling_reconstruction")
        updates.update({data: res[-1][-1]})
        gibbs_sampling = theano.function([], res, updates=updates)

        reconstructions = []
        for i in xrange(k_batch):
            result = gibbs_sampling()
            [_, _, _, _, reconstruction_chain, _] = result
            reconstructions.append(reconstruction_chain[-1])

        if self.track_progress:
            self.track_progress.visualise_reconstructions(orig, reconstructions, plot_n)

        return reconstructions[-1]

    def reconstruct_association(self, x, y=None, k=1, bit_p=0, plot_n=None, plot_every=1):
        # Initialise parameters
        if 'Tensor' not in str(type(x)):
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
            self.track_progress.visualise_reconstructions(x.get_value(borrow=True), reconstructions, plot_n)

        return reconstruction_chain[-1]


class AssociativeRBM(RBM):
    pass


class GaussianRBM(RBM):
    '''
    Implements an RBM with Gausian Visible Units with noisy rectifier hidden unit
    Must set mean = 0 and variance = 1 for dataset you use
    '''

    def __init__(self,v_n=10,
                 v_n2=10,
                 h_n=10,
                 associative=False,
                 W=None,
                 U=None,
                 h_bias=None,
                 v_bias=None,
                 v_bias2=None,
                 visible_unit = GaussianVisibleUnit,
                 hidden_unit = NReLUnit,
                 h_activation_fn=log_sig,
                 v_activation_fn=log_sig,
                 v_activation_fn2=log_sig,
                 cd_type=PERSISTENT,
                 cd_steps=1,
                 train_parameters=None,
                 progress_logger=None):

        RBM.__init__(self, v_n, v_n2, h_n, associative, W, U, h_bias, v_bias, v_bias2,
                     h_activation_fn, v_activation_fn, v_activation_fn2,
                     cd_type, cd_steps, train_parameters, progress_logger)

        self.variance = 1
        self.visible_unit = GaussianVisibleUnit()
        self.visible_unit = visible_unit()
        self.hidden_unit = hidden_unit(self.h_n)

    def __str__(self):
        return "gaussian_" + RBM.__str__(self)

    def calc_free_energy(self, v, v2=None):
        w = self.W
        v_bias = self.v_bias
        h_bias = self.h_bias

        t0 = self.visible_unit.energy(v, v_bias)
        t1 = T.dot(v, w) + h_bias

        if v2:
            u = self.U
            v_bias2 = self.v_bias2
            # t0 += -T.dot(v2, v_bias2) # For classRBM
            t0 += self.visible_unit.energy(v2, v_bias2)  # For GaussianUnits
            t1 += T.dot(v2, u)

        t2 = - T.sum(T.log(1 + T.exp(t1)))

        return t0 + t2

    def prop_up(self, v, v2=None):
        """Propagates v to the hidden layer. """
        h_in = T.dot(v, self.W) + self.h_bias
        if np.any(v2):
            h_in += T.dot(v2, self.U)  # Associative Layer

        return [h_in, (self.hidden_unit.scale(h_in))]

    def __prop_down(self, h, connectivity, bias, v_unit):
        """Propagates h to the visible layer. """
        v_in = T.dot(h, connectivity.T) + bias
        return [v_in, (v_unit.scale(v_in))]

    def prop_down(self, h):
        return self.__prop_down(h, self.W, self.v_bias, self.visible_unit)

    def prop_down_assoc(self, h):
        return self.__prop_down(h, self.U, self.v_bias2, self.visible_unit2)

    def sample_h_given_v(self, v, v2=None):
        h_total_input, h_p_activation = self.prop_up(v, v2)
        h_sample = self.hidden_unit.activate(h_p_activation)
        return [h_total_input, h_p_activation, h_sample]

    def __sample_v_given_h(self, h, prop_down_fn):
        v_total_input, v_p_activation = prop_down_fn(h)
        v_sample = self.visible_unit.activate(v_p_activation)
        return [v_total_input, v_p_activation, v_sample]

    def sample_v_given_h(self, h_sample):
        return self.__sample_v_given_h(h_sample, self.prop_down)

    def sample_v_given_h_assoc(self, h_sample):
        return self.__sample_v_given_h(h_sample, self.prop_down) + self.__sample_v_given_h(h_sample, self.prop_down_assoc)


def test_rbm():
    print "Testing RBM"

    # Load mnist hand digits
    datasets = loader.load_digits(n=[100, 0, 100], digits=[1])
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
    rbm.plot_samples(test_set_x)

    # Store Parameters
    rbm.save()

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded:
        print "... loaded trained RBM"

    # Move back to root
    os.chdir(root_dir)
    print "... moved to " + root_dir

if __name__ == '__main__':
    test_rbm()
