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
        self.n_chains = n_chains
        self.n_samples = n_samples
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
        self.plot_epoch_image = plot_epoch_image
        self.plot_sample = plot_sample

    def __str__(self):
        return "epoch" + str(self.epochs) + \
               "_batch" + str(self.batch_size) + \
               "_lr" + str(self.learning_rate) + \
               "_" + self.momentum_type + str(self.momentum) + \
               ("_sparsity"
                + "_t" + str(self.sparsity_target)
                + "_c" + str(self.sparsity_cost) +
                "_d" + str(self.sparsity_decay) if self.sparsity_constraint else "")


class RBM(object):

    def __init__(self, 
                 v_n,
                 h_n,
                 W=None,
                 h_bias=None,
                 v_bias=None,
                 h_activation_fn=log_sig,
                 v_activation_fn=log_sig,
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
        self.v_activation_fn = v_activation_fn
        # Hidden Layer
        self.h_n = h_n
        self.h_bias = h_bias
        self.h_activation_fn = h_activation_fn
        # Gibbs Sampling Method
        self.cd_type = cd_type
        self.cd_steps = cd_steps

        self.params = [self.W, self.v_bias, self.h_bias]

        print "Initialised"

    def __str__(self):
        return "RBM_" + str(self.h_n)

    def free_energy(self, v):        
        wv_c = T.dot(v, self.W) + self.h_bias
        return - T.dot(v, self.v_bias) - T.sum(T.log(1 + T.exp(wv_c)))

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

    def get_reconstruction_cost(self, x, v_total_inputs):
        p = self.v_activation_fn(v_total_inputs)
        cross_entropy = T.mean(
            T.sum(
                x * T.log(p) + (1 - x) * T.log(1 - p),
                axis=1
            )
        )
        return cross_entropy

    def get_cost_updates(self, x, param_increments):
        param = self.train_parameters
        # Cast parameters
        lr = T.cast(param.learning_rate, dtype=t_float_x)
        m = T.cast(param.momentum, dtype=t_float_x)
        weight_decay = T.cast(param.weight_decay, dtype=t_float_x)
        old_DW, old_Dvbias, old_Dhbias = param_increments[:-1]

        # Perform Gibbs Sampling to generate negative statistics
        res = self.contrastive_divergence(x, self.cd_steps)
        updates = res[0]
        v_sample = res[1]
        v_total_inputs = res[2]

        # Differentiate cost function w.r.t params to get gradients for param updates
        cost = T.mean(self.free_energy(x)) - T.mean(self.free_energy(v_sample))
        g_W, g_v, g_h = T.grad(cost, self.params, consider_constant=[v_sample])

        # Classical Momentum - From Sutskever, Hinton.
        # v_new = momentum * v_old + lr * grad_wrt_w
        # w_new = w_old + v_new
        new_DW = m * old_DW - lr * g_W
        new_Dhbias = m * old_Dhbias - lr * g_h
        new_Dvbias = m * old_Dvbias - lr * g_v
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

        cross_entropy = self.get_reconstruction_cost(x, v_total_inputs)

        return cross_entropy, updates

        # Nesterov update:
        # v_new = m * v_old - lr * Df(x_old + m*v_old)
        # x_new = x_old + v_new
        # <=> x_new = [x_old + m * v_old] - lr * Df([x_old + m * v_old])
    def get_nesterov_cost_updates(self,
                                  lr=0.1,
                                  m=0.5,
                                  weight_decay=0.001,
                                  sparsity_target=0.01,
                                  sparsity_cost=0.01,
                                  k=1):
        lr = T.cast(lr, dtype=theano.config.floatX)
        m = T.cast(m, dtype=theano.config.floatX)
        weight_decay = T.cast(weight_decay, dtype=theano.config.floatX)

        partial_updates = []
        # {self.W: self.W + m * self.old_DW,
        #                    self.h_bias: self.h_bias + m * self.old_Dhbias,
        #                    self.v_bias: self.v_bias + m * self.old_Dvbias}

        partial_updates.append((self.W, self.W + m * self.old_DW))
        partial_updates.append((self.h_bias, self.h_bias + m * self.old_Dhbias))
        partial_updates.append((self.v_bias, self.v_bias + m * self.old_Dvbias))

        updates, v_sample, pre_sigmoid_nv = self.contrastive_divergence(k)

        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(v_sample))

        # Computes gradient for the cost function for parameter updates
        g_W, g_v ,g_h = T.grad(cost, self.params, consider_constant=[v_sample])

        new_DW = m * self.old_DW - lr * g_W
        new_Dhbias = m * self.old_Dhbias - lr * g_h
        new_Dvbias = m * self.old_Dvbias - lr * g_v
        # new_W = self.W + new_DW
        # new_hbias = self.h_bias + new_Dhbias
        # new_vbias = self.v_bias + new_Dvbias
        new_W = self.W - lr * g_W
        new_hbias = self.h_bias - lr * g_h
        new_vbias = self.v_bias - lr * g_v
        # penalise weight decay. W is set as W+m*v, we need to subtract m*v to get old W)
        new_W -= lr * weight_decay * (self.W - m * self.old_DW)
        new_hbias -= lr * weight_decay * (self.h_bias - m * self.old_Dhbias)
        new_vbias -= lr * weight_decay * (self.v_bias - m * self.old_Dvbias)
        # update parameters
        updates[self.W] = new_W
        updates[self.h_bias] = new_hbias
        updates[self.v_bias] = new_vbias
        # update velocities
        updates[self.old_DW] = new_DW
        updates[self.old_Dhbias] = new_Dhbias
        updates[self.old_Dvbias] = new_Dvbias

        cross_entropy = self.get_reconstruction_cost(updates, pre_sigmoid_nv)

        return cross_entropy, updates, partial_updates

    def get_train_fn(self, train_data):
        param = self.train_parameters
        batch_size = param.batch_size
        index = T.lscalar()
        x = T.matrix('x')

        # Initialise Variable used for training
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

        if param.momentum_type is 'nesterov':
            cross_entropy, updates, partial_updates = self.get_nesterov_cost_updates()
    
            weights_partial_update = theano.function(
                inputs=[],
                outputs=[],
                updates=partial_updates
            )
            train_function = theano.function(
                inputs=[index],
                outputs=cross_entropy,
                updates=updates,
                givens={
                    x: train_data[index * batch_size: (index + 1) * batch_size],
                }
            )
    
            def train_rbm(i):
                weights_partial_update()
                return train_function(i)
    
        else:
            cross_entropy, updates = self.get_cost_updates(x, param_increments)
            train_rbm = theano.function(
                [index],
                cross_entropy,  # use cross entropy to keep track
                updates=updates,
                givens={
                    x: train_data[index * batch_size: (index + 1) * batch_size]
                },
                name='train_rbm'
            )
        
        return train_rbm

    def train(self, train_data):
        """Trains RBM. For now, input needs to be Theano matrix"""

        param = self.train_parameters
        batch_size = param.batch_size
        mini_batches = train_data.get_value(borrow=True).shape[0] / batch_size

        train_fn = self.get_train_fn(train_data)

        if param.plot_epoch_image:
            if not param.output_directory:
                out_dir = '_'.join([str(self), str(param)])
            else:
                out_dir = param.output_directory

            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            os.chdir(out_dir)

        plotting_time = 0.
        start_time = time.clock()       # Measure training time
        for epoch in xrange(param.epochs):
            mean_cost = []
            for batch_index in xrange(mini_batches):
                mean_cost += [train_fn(batch_index)]
    
            print 'Epoch %d, cost is ' % epoch, np.mean(mean_cost)
    
            if param.plot_epoch_image:
                plotting_start = time.clock()       # Measure plotting time 
                image = Image.fromarray(
                    tile_raster_images(
                        X=self.W.get_value(borrow=True).T,
                        img_shape=(28, 28),
                        tile_shape=(10, 10),
                        tile_spacing=(1, 1)
                    )
                )
                image.save('epoch_%i.png' % epoch)
                plotting_stop = time.clock()
                plotting_time += (plotting_stop - plotting_start)
    
        end_time = time.clock()
        pre_training_time = (end_time - start_time) - plotting_time
    
        print ('Training took %f minutes' % (pre_training_time / 60.))
        os.chdir('../')

        return [self.W, self.v_bias, self.h_bias]

    def retrieve_parameters(self):
        return [self.W, self.v_bias, self.h_bias, self.train_parameters]


def test_rbm():
    print "Testing RBM"

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 100

    tr = TrainParam(momentum_type='classical')

    rbm = RBM(n_visible, n_hidden, train_parameters=tr)
    rbm.train(test_set_x)

if __name__ == '__main__':
    test_rbm()