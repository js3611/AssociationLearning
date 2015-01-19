import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time

try:
    import PIL.Image as Image
except ImportError:
    import Image

import os

from utils import tile_raster_images
from utils import load_data


class RBM(object):
    """Restricted Boltzmann Machine Implementation"""

    def __init__(
            self,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            hbias=None,
            vbias=None
    ):
        """RBM Constructor which initialises the parameters.

        Keyword arguments:
        input --
        n_visible -- number of visible units (default 0)
        n_hidden -- number of hidden units (default 0)
        W -- weight matrix between visible and hidden layers (default zero matrix)
        vbias -- bias for visible layer (default None)
        hbias -- bias for hidden layer (default None)
        """

        # Random Number Generator
        numpy_rng = np.random.RandomState(1234)
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if vbias is None:
            vbias = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                name='vbias', borrow=True
            )

        if hbias is None:
            hbias = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='hbias', borrow=True
            )

        self.input = input if input else T.matrix('input')
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng                           # Bad idea to assign random generator
        # For momentum
        self.old_DW = theano.shared(value=np.zeros_like(initial_W), name='old_DW', borrow=True)
        self.old_Dhbias = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX), name='old_Dhbias', borrow=True)
        self.old_Dvbias = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX), name='old_Dvbias', borrow=True)
        self.params = [self.W, self.hbias, self.vbias]
        self.all_params = [self.W, self.hbias, self.vbias, self.old_DW, self.old_Dhbias, self.old_Dvbias]
        print "RBM initialised"

    def free_energy(self, v):
        '''Returns Theano expression for free energy of input vector v'''
        wv_c = T.dot(v, self.W) + self.hbias
        return - T.dot(v, self.vbias) - T.sum(T.log(1 + T.exp(wv_c)))

    def propup(self, vis):
        '''Computes the probabilities of each hidden unit being 'on' by applying sigmoid function to the input visible units transformed by weights'''
        pre_sigm = T.dot(vis, self.W) + self.hbias
        return [pre_sigm, T.nnet.sigmoid(pre_sigm)]

    def propdown(self,h):
        '''Computes the probabilities of each visible unit being 'on' by applying sigmoid function to the input hidden units transformed by weights'''
        pre_sigm = T.dot(h, self.W.T) + self.vbias
        return [pre_sigm, T.nnet.sigmoid(pre_sigm)]

    def sample_h_given_v(self, v_sample):
        '''Generate hidden sample using visible sample. Each hidden unit will be 'switched on' with probability given by propup'''
        pre_sigm, h_mean = self.propup(v_sample)
        h_sample = self.theano_rng.binomial(size=h_mean.shape,
                                            n=1, p=h_mean,
                                            dtype=theano.config.floatX)
        return [pre_sigm, h_mean, h_sample]

    def sample_v_given_h(self, h_sample):
        '''Generate visible sample using visible sample. Each visible unit will be 'switched on' with probability given by propdown'''
        pre_sigm, v_mean = self.propdown(h_sample)
        v_sample = self.theano_rng.binomial(size=v_mean.shape,
                                            n=1, p=v_mean,
                                            dtype=theano.config.floatX)
        return [pre_sigm, v_mean, v_sample]

    def gibbs_hvh(self, h_sample):
        pre_sigm_v, v_mean, v_sample = self.sample_v_given_h(h_sample)
        pre_sigm_h, h_mean, h_sample = self.sample_h_given_v(v_sample)
        return [pre_sigm_v, v_mean, v_sample,
                pre_sigm_h, h_mean, h_sample]

    def gibbs_vhv(self, v_sample):
        pre_sigm_h, h_mean, h_sample = self.sample_h_given_v(v_sample)
        pre_sigm_v, v_mean, v_sample = self.sample_v_given_h(h_sample)
        return [pre_sigm_h, h_mean, h_sample,
                pre_sigm_v, v_mean, v_sample]


    def contrastive_divergence(self, k=1):
        pre_sigm_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        chain_start = ph_sample
        # Chain starting at h_sample
        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
        )
        chain_end = nv_samples[-1]

        # prob better return everything that scan returns
        return [updates, chain_end, pre_sigmoid_nvs]

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

    def get_cost_updates(self, lr=0.1, m=0.5, k=1):
        updates, v_sample, pre_sigmoid_nv = self.contrastive_divergence(k)

        cost = T.mean(self.free_energy(self.input)) - T.mean(
            self.free_energy(v_sample))

        # Cast meta parameters
        lr = T.cast(lr, dtype=theano.config.floatX)
        m = T.cast(m, dtype=theano.config.floatX)

        # Computes gradient for the cost function for parameter updates
        g_W, g_h, g_v = T.grad(cost, self.params, consider_constant=[v_sample])
        # From Sutskever, Hinton.
        # v_new = momentum * v_old + lr * grad_wrt_w
        # w_new = w_old + v_new
        new_DW = m * self.old_DW - lr * g_W
        new_Dhbias = m * self.old_Dhbias - lr * g_h
        new_Dvbias = m * self.old_Dvbias - lr * g_v
        new_W = self.W + new_DW
        new_hbias = self.hbias + new_Dhbias
        new_vbias = self.vbias + new_Dvbias
        # update parameters
        updates[self.W] = new_W
        updates[self.hbias] = new_hbias
        updates[self.vbias] = new_vbias
        # update velocities
        updates[self.old_DW] = new_DW
        updates[self.old_Dhbias] = new_Dhbias
        updates[self.old_Dvbias] = new_Dvbias

        # without momentum
        gparams = T.grad(cost, self.params, consider_constant=[v_sample])
        for gparam, param in zip(gparams, self.params):
            new_param = param - gparam * T.cast(lr, dtype=theano.config.floatX)
            updates[param] = new_param

        cross_entropy = self.get_reconstruction_cost(updates, pre_sigmoid_nv)

        return cross_entropy, updates

##################################################
#               Training RBM                     #
##################################################

# compute_test_value is 'off' by default, meaning this feature is inactive
# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'

def test_rbm(learning_rate=0.1,
             momentum=0.5,
             training_epochs=15,
             dataset='mnist.pkl.gz',
             batch_size=20,
             n_chains=20,
             n_samples=10,
             output_folder='rbm_plots',
             n_hidden=500):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar() # index to a minibatch
    x = T.matrix('x') # to represent data as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialise RBM
    rbm = RBM(input=x, n_visible=28*28, n_hidden=n_hidden)

    # we are using cross_entropy as proxy to "log-likelihood"
    # we want to minimise nll but we cannot measure it because of Z. we use cross_entropy to approximate it
    cross_entropy, updates = rbm.get_cost_updates(lr=learning_rate, m=momentum, k=1)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)


    train_rbm = theano.function(
        [index],
        cross_entropy, # use cross entropy to keep track
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = time.clock()

    print "Start Training"

    for epoch in xrange(training_epochs):
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, np.mean(mean_cost)

        plotting_start = time.clock()

        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))


    # Sampling from the RBM #
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    test_idx = rng.randint(number_of_test_samples - n_chains)

    persistent_vis_chain = theano.shared(
        np.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
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
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    updates.update({persistent_vis_chain: vis_samples[-1]})

    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )

    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')

if __name__ == '__main__':

    test_rbm(learning_rate=0.1,
             momentum=0.9,
             training_epochs=15,
             dataset='mnist.pkl.gz',
             batch_size=20,
             n_chains=20,
             n_samples=10,
             output_folder='rbm_plots10momentum',
             n_hidden=10)


    # test_rbm(learning_rate=0.1, training_epochs=15,
    #          dataset='mnist.pkl.gz', batch_size=20,
    #          n_chains=20, n_samples=10, output_folder='rbm_plots500',
    #          n_hidden=500)