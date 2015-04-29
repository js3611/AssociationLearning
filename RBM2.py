import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
from utils import load_data

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

# Theano Debugging Configuration
# compute_test_value is 'off' by default, meaning this feature is inactive
# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature
#theano.config.optimizer = 'None'
#theano.config.exception_verbosity = 'high'


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
        self.params = [self.W, self.v_bias, self.h_bias]

        print "... initialised RBM"

    def __str__(self):
        return "rbm_" + str(self.h_n) + \
               "_" + self.cd_type + str(self.cd_steps) + \
               "_" + str(self.train_parameters)

    def free_energy(self, v, w=None, v_bias=None, h_bias=None):
        if not w:
            w = self.W
        if not v_bias:
            v_bias = self.v_bias
        if not h_bias:
            h_bias = self.h_bias

        wv_c = T.dot(v, w) + h_bias
        return - T.dot(v, v_bias) - T.sum(T.log(1 + T.exp(wv_c)))

    def prop_up(self, v):
        """Propagates v to the hidden layer. """
        h_total_input = T.dot(v, self.W) + self.h_bias
        h_p_activation = self.h_activation_fn(h_total_input)
        return [h_total_input, h_p_activation]

    def prop_down(self, h):
        """Propagates h to the visible layer. """
        v_total_input = T.dot(h, self.W.T) + self.v_bias
        v_p_activation = self.v_activation_fn(v_total_input)
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

    def negative_statistics(self, x):
        if self.cd_type is PERSISTENT:
            return self.pcd(self.cd_steps)
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

    def get_cost_updates(self, x, param_increments):
        param = self.train_parameters
        # Cast parameters
        lr = T.cast(param.learning_rate, dtype=t_float_x)
        m = T.cast(param.momentum, dtype=t_float_x)
        weight_decay = T.cast(param.weight_decay, dtype=t_float_x)

        # Declare parameter update variables
        old_DW, old_Dvbias, old_Dhbias = param_increments[:-1]
        pre_updates = []

        if param.momentum_type is NESTEROV:
            pre_updates.append((self.W, self.W + m * old_DW))
            pre_updates.append((self.h_bias, self.h_bias + m * old_Dhbias))
            pre_updates.append((self.v_bias, self.v_bias + m * old_Dvbias))

        # Perform Gibbs Sampling to generate negative statistics
        res = self.negative_statistics(x)
        updates = res[0]
        v_sample = res[1]
        v_total_inputs = res[2]

        # Differentiate cost function w.r.t params to get gradients for param updates
        cost = T.mean(self.free_energy(x)) - T.mean(self.free_energy(v_sample))
        g_W, g_v, g_h = T.grad(cost, self.params, consider_constant=[v_sample])

        new_DW = m * old_DW - lr * g_W
        new_Dhbias = m * old_Dhbias - lr * g_h
        new_Dvbias = m * old_Dvbias - lr * g_v

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

            # Weight Decay
            new_W -= lr * weight_decay * self.W
            new_hbias -= lr * weight_decay * self.h_bias
            new_vbias -= lr * weight_decay * self.v_bias

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
        # update velocities
        updates[old_DW] = new_DW
        updates[old_Dhbias] = new_Dhbias
        updates[old_Dvbias] = new_Dvbias

        if self.cd_type is PERSISTENT:
            # cost = self.get_reconstruction_cost(x, v_total_inputs)
            measure_cost = self.get_pseudo_likelihood(x, updates)
        else:
            measure_cost = self.get_reconstruction_cost(x, v_total_inputs)

        return measure_cost, updates, pre_updates

    def get_train_fn(self, train_data):
        param = self.train_parameters
        batch_size = param.batch_size
        index = T.lscalar()
        x = T.matrix('x')

        # Initialise Variables used for training
        # For momentum
        old_DW = theano.shared(value=np.zeros(self.W.get_value().shape, dtype=t_float_x),
                               name='old_DW', borrow=True)
        old_Dvbias = theano.shared(value=np.zeros(self.v_n, dtype=t_float_x),
                                   name='old_Dvbias', borrow=True)
        old_Dhbias = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
                                   name='old_Dhbias', borrow=True)

        # For sparsity cost
        active_probability_h = theano.shared(value=np.zeros(self.h_n, dtype=t_float_x),
                                             name="active_probability_h",
                                             borrow=True)

        param_increments = [old_DW, old_Dvbias, old_Dhbias, active_probability_h]

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

        def train_fn(i):
            pre_train()
            return train_rbm(i)

        return train_fn

    def create_and_move_to_output_dir(self):
        param = self.train_parameters
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

    def train(self, train_data):
        """Trains RBM. For now, input needs to be Theano matrix"""

        param = self.train_parameters
        batch_size = param.batch_size
        mini_batches = train_data.get_value(borrow=True).shape[0] / batch_size

        train_fn = self.get_train_fn(train_data)
        self.create_and_move_to_output_dir()

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
        os.chdir('../../')

        return [mean_cost]

    def classify(self, data):

        # obtain from rbm

        # input x, get y out

        # use argmax

        return 0

    def plot_samples(self, test_data):
        self.create_and_move_to_output_dir()

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
        os.chdir('../..')

    def save(self):
        self.create_and_move_to_output_dir()
        datastorage.store_object(self)
        os.chdir('../..')
        print "... saved RBM object to " + os.getcwd() + "/data/" + str(self)


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
                    weight_decay=0.001,
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_cost=0.01,
                    sparsity_decay=0.1,
                    plot_during_training=True)

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 10

    rbm = RBM(n_visible,
              n_hidden,
              cd_type=PERSISTENT,
              cd_steps=1,
              train_parameters=tr)

    # Train RBM
    rbm.train(train_set_x)

    # Test RBM
    rbm.plot_samples(test_set_x)

    # Store Parameters
    rbm.save()

    # Load RBM (test)
    rbm.create_and_move_to_output_dir()
    loaded = datastorage.retrieve_object(str(rbm))
    print loaded


def get_target_vector(x):
    xs = np.zeros(10, dtype=np.int)
    xs[x] = 1
    return xs


def test_rbm_association():
    print "Testing Associative RBM"

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Reformat the train label
    print test_set_y
    new_train_set_y = np.matrix(map(lambda x: get_target_vector(x), train_set_y.eval()))
    # # Cast to int type
    # train_set_y = T.cast(theano.shared(new_train_set_y), 'int32')
    train_set_y = theano.shared(new_train_set_y)
    print train_set_y

    # Combine the input
    train_set_xy = T.concatenate([train_set_x, train_set_y], 1)
    print train_set_xy
    print train_set_xy.eval().shape


    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.01,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.001,
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_cost=0.01,
                    sparsity_decay=0.1,
                    plot_during_training=True,
                    output_directory="AssociationTest")

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 10

    rbm = RBM(n_visible,
              n_hidden,
              cd_type=PERSISTENT,
              cd_steps=1,
              train_parameters=tr)

    # Train RBM
    # rbm.train(train_set_x)

    # Test RBM
    # rbm.plot_samples(test_set_x)

    # Store Parameters
    # rbm.save()

    # Load RBM (test)
    # rbm.create_and_move_to_output_dir()
    # loaded = datastorage.retrieve_object(str(rbm))
    # print loaded


if __name__ == '__main__':
    test_rbm_association()
    # test_rbm()