import copy
import datastorage as store
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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
        self.untied = False
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

        if type(tr) is not list:
            tr = [tr for i in xrange(self.n_layers)]

        if type(rbm_configs) is not list:
            rbm_configs = [rbm_configs for i in xrange(self.n_layers)]

        for i in xrange(self.n_layers):
            rbm_config = rbm_configs[i]
            rbm_config.v_n = topology[i]
            rbm_config.h_n = topology[i + 1]
            # rbm_config.training_parameters = tr[i]  # Ensure it has parameters
            rbm_layer = RBM(rbm_config)
            self.rbm_layers.append(rbm_layer)

    def __str__(self):
        return 'dbn_' + str(self.n_layers) + \
               'lys_' + '_'.join([str(i) for i in self.topology])

    def pretrain(self, train_data, cache=False, train_further=False, names=None):
        if type(cache) is not list:
            cache = np.repeat(cache, self.n_layers)
        if type(train_further) is not list:
            train_further = np.repeat(train_further, self.n_layers)

        layer_input = train_data
        for i in xrange(len(self.rbm_layers)):
            rbm = self.rbm_layers[i]
            print '... training layer {}, {}'.format(i, rbm)

            self.data_manager.move_to('{}/layer/{}/{}'.format(self.out_dir, i, rbm))

            # Check Cache
            cost = 0
            name=names[i] if names else str(rbm)
            ret_name = cache[i] if isinstance(cache[i], str) else name
            loaded = store.retrieve_object(ret_name)
            if cache[i] and loaded:
                # TODO override neural network's weights too
                epochs = rbm.config.train_params.epochs
                rbm = loaded
                rbm.config.train_params.epochs = epochs
                # Override the reference
                self.rbm_layers[i] = rbm
                print "... loaded trained layer {}".format(ret_name)

                if train_further[i]:
                    cost += np.mean(rbm.train(layer_input))
                    self.data_manager.persist(rbm, name=name)
            else:
                if rbm.train_parameters.sparsity_constraint:
                    rbm.set_initial_hidden_bias()
                    rbm.set_hidden_mean_activity(layer_input)
                cost += np.mean(rbm.train(layer_input))
                self.data_manager.persist(rbm, name=name)

            self.data_manager.move_to_project_root()
            # os.chdir('../..')

            # Pass the input through sampler method to get next layer input
            sampled_layer = rbm.sample_h_given_v(layer_input)
            transform_input = sampled_layer[2]
            f = theano.function([], transform_input)
            res = f()
            layer_input = theano.shared(res)
        return cost

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
            rbm = self.rbm_layers[i] if not self.untied else self.inference_layers[i]
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
            rbm = self.rbm_layers[i] if not self.untied else self.generative_layers[i]
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
            top_x = self.bottom_up_pass(x, 0, self.n_layers - 1)
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
                recon = self.top_down_pass(reconstruction_chain[-1], self.n_layers - 1)
            else:
                recon = reconstruction_chain[-1]
            reconstructions.append(recon)

        if self.rbm_layers[0].track_progress:
            self.rbm_layers[0].track_progress.visualise_reconstructions(orig,
                                                                        reconstructions,
                                                                        plot_n,
                                                                        img_name)

        return reconstructions[-1]

    def sample(self, n=1, k=10, rand_type='noisy_mean'):

        top_layer = self.rbm_layers[-1]

        if self.n_layers > 1 and rand_type in ['mean', 'noisy_mean']:
            print 'sampling according to active_probability of layer below'
            pen = self.rbm_layers[-2]
            active_h = pen.active_probability_h.get_value(borrow=True)
            x = top_layer.sample(n, k, rand_type=rand_type, p=active_h)
        else:
            # Sample between top two layers
            x = top_layer.sample(n, k, rand_type=rand_type, p=0.5)

        # prop down the output to visible unit if it is not RBM
        if self.n_layers > 1:
            sampled = self.top_down_pass(x, self.n_layers - 1)
        else:
            sampled = x

        return sampled

    def untie_weights(self, include_top=False):
        # Untie all the weights except for the top layer
        self.untied = True
        layers = self.rbm_layers if include_top else self.rbm_layers[:-1]
        for i, rbm in enumerate(layers):
            W = rbm.W.get_value(borrow=False)
            h_bias = rbm.h_bias.get_value(borrow=False)
            v_bias = rbm.v_bias.get_value(borrow=False)
            W1 = theano.shared(W, name=('inference_W_%d' % i))
            v1 = theano.shared(v_bias, name=('inference_vbias_%d' % i))
            h1 = theano.shared(h_bias, name=('inference_hbias_%d' % i))
            W2 = theano.shared(copy.deepcopy(W), name=('generative_W_%d' % i))
            v2 = theano.shared(copy.deepcopy(v_bias), name=('generative_vbias_%d' % i))
            h2 = theano.shared(copy.deepcopy(h_bias), name=('generative_hbias_%d' % i))
            self.inference_layers.append(RBM(config=rbm.config, W=W1, h_bias=h1, v_bias=v1))
            self.generative_layers.append(RBM(config=rbm.config, W=W2, h_bias=h2, v_bias=v2))

    def save_untied_weights(self, name):
        manager = self.data_manager
        # make directory
        for t in ['generative', 'inference']:
            for i in xrange(len(self.generative_layers)):
                rbm = self.generative_layers[i] if t == 'generative' else self.inference_layers[i]
                manager.persist(rbm.W.get_value(borrow=False), name=('%s_W' % rbm), out_dir=('%s_weights/%d' % t, i))
                manager.persist(rbm.v_bias.get_value(borrow=False), name=('%s_vbias' % rbm),
                                out_dir=('%s_weights/%d' % t, i))
                manager.persist(rbm.h_bias.get_value(borrow=False), name=('%s_hbias' % rbm),
                                out_dir=('%s_weights/%d' % t, i))

    def retrieve_untied_weights(self):
        manager = self.data_manager
        if self.untied:
            self.untie_weights()
        # make directory
        for t in ['generative', 'inference']:
            for i in xrange(len(self.generative_layers)):
                rbm = self.generative_layers[i] if t == 'generative' else self.inference_layers[i]
                W = manager.retrieve(('%s_W' % rbm), out_dir=('%s_weights/%d' % t, i))
                v_bias = manager.retrieve(('%s_vbias' % rbm), out_dir=('%s_weights/%d' % t, i))
                h_bias = manager.retrieve(('%s_hbias' % rbm), out_dir=('%s_weights/%d' % t, i))
                rbm.W.set_value(rbm.get_initial_weight(W, W.shape[0], W.shape[1], '%s_W_%d'))
                rbm.v_bias.set_value(rbm.get_initial_bias(v_bias, len(v_bias), ('%s_vbias_%d' % t, i)))
                rbm.h_bias.set_value(rbm.get_initial_weight(h_bias, len(h_bias), ('%s_hbias_%d' % t, i)))

    def wake_phase(self, data):
        # PERFORM A BOTTOM-UP PASS TO GET WAKE/POSITIVE PHASE
        # PROBABILITIES AND SAMPLE STATES
        layers = self.inference_layers
        wake_probs = []
        wake_states = []
        layer_in = data
        for rbm in layers:
            _, wake_p, wake_s = rbm.sample_h_given_v(layer_in)
            wake_probs.append(wake_p)
            wake_states.append(wake_s)
            layer_in = wake_s

        return wake_probs, wake_states

    def sleep_phase(self, pen_in):
        # STARTING FROM THE END OF THE GIBBS SAMPLING RUN, PERFORM A
        # TOP-DOWN GENERATIVE PASS TO GET SLEEP/NEGATIVE PHASE
        # PROBABILITIES AND SAMPLE STATES
        layers = self.generative_layers
        sleep_probs = []
        sleep_states = [pen_in]
        for rbm in reversed(layers):
            _, sleep_p, sleep_s = rbm.sample_v_given_h(sleep_states[-1])
            sleep_probs.append(sleep_p)
            sleep_states.append(sleep_s)

        return sleep_probs, sleep_states

    def fine_tune_cd(self, wake_state):
        # CONTRASTIVE DIVERGENCE AT TOP LAYER
        top_rbm = self.rbm_layers[-1]
        pen_state = wake_state
        [updates, chain_end, _, _, _, _, _, _, _] = top_rbm.negative_statistics(pen_state)
        cost = T.mean(top_rbm.free_energy(pen_state)) - T.mean(top_rbm.free_energy(chain_end))
        grads = T.grad(cost, top_rbm.params, consider_constant=[chain_end])
        lr = top_rbm.train_parameters.learning_rate
        for (p, g) in zip(top_rbm.params, grads):
            # TODO all the special updates like momentum
            updates[p] = p - lr * g

        return chain_end, updates

    def get_predictions(self, data, sleep_probs, sleep_states, wake_states):
        psleep_states = []  # [hid_prob, pen_prob, ...]
        pwake_states = []  # [vis_prob, hid_prob, ... ]
        s = sleep_states[:-1]
        s.reverse()
        rev_sleep_in = [sleep_probs[-1]] + s # [sl_v_prob, sl_hid_state, sl_pen_state]
        for rbm, sleep_in in zip(self.inference_layers, rev_sleep_in):
            _, p = rbm.prop_up(sleep_in)
            psleep_states.append(p)
        for rbm, wake_in in zip(reversed(self.generative_layers), reversed(wake_states)):
            _, p = rbm.prop_down(wake_in)
            pwake_states.append(p)
        # wake_states.reverse()
        pwake_states.reverse()
        wake_states = [data] + wake_states  # [data, hid_state, pen_state, ...]
        sleep_states = rev_sleep_in  # [vis_prob, hid_state, pen_state, ...]
        return psleep_states, pwake_states, sleep_states, wake_states

    def update_generative_weights(self, batch_size, pwake_states, wake_states, updates):
        # UPDATES TO GENERATIVE PARAMETERS
        for i in xrange(len(self.generative_layers)):
            rbm = self.generative_layers[i]
            r = rbm.train_parameters.learning_rate
            wake_state = wake_states[i]
            pwake_state = pwake_states[i]
            statistics_diff = wake_state - pwake_state
            updates[rbm.W] = rbm.W + r * T.dot(wake_states[i + 1].T, statistics_diff).T / batch_size
            updates[rbm.v_bias] = rbm.v_bias + r * T.mean(statistics_diff, axis=0)

        return updates

    def update_inference_weights(self, batch_size, psleep_states, sleep_states, updates):
        for i in xrange(0, len(self.inference_layers)):
            rbm = self.inference_layers[i]
            r = rbm.train_parameters.learning_rate
            sleep_state = sleep_states[i + 1]
            psleep_state = psleep_states[i]
            statistics_diff = sleep_state - psleep_state
            updates[rbm.W] = rbm.W + r * T.dot(sleep_states[i].T, statistics_diff) / batch_size
            updates[rbm.h_bias] = rbm.h_bias + r * T.mean(statistics_diff, axis=0)

        return updates

    def get_fine_tune_updates(self, data, batch_size):
        '''
        Fine tunes DBN. Inference weights and Generative weights will be untied
        '''

        # WAKE-PHASE [hid_prob, pen_prob, ...], [hid_state, pen-state,...]
        wake_probs, wake_states = self.wake_phase(data)

        # TOP LAYER CD
        chain_end, updates = self.fine_tune_cd(wake_states[-1])

        # SLEEP PHASE: [hid_prob, vis_prob], [pen_state, hid_state, vis_state] ...
        sleep_probs, sleep_states = self.sleep_phase(chain_end)

        # Prediction
        psleep_states, pwake_states, sleep_states, wake_states = self.get_predictions(data, sleep_probs, sleep_states,
                                                                                      wake_states)

        # UPDATES TO GENERATIVE PARAMETERS
        updates = self.update_generative_weights(batch_size, pwake_states, wake_states, updates)

        # UPDATES TO INFERENCE PARAMETERS
        updates = self.update_inference_weights(batch_size, psleep_states, sleep_states, updates)

        return updates

    def fine_tune(self, data, epochs=5, batch_size=10):
        print '... fine tuning'

        if not self.untied:
            self.untie_weights()

        mini_batches = data.get_value(borrow=True).shape[0] / batch_size

        for epoch in xrange(epochs):
            print '... epoch %d' % epoch

            i = T.iscalar()
            x = T.matrix('x')
            updates = self.get_fine_tune_updates(x, batch_size)
            fine_tune = theano.function([i], [], updates=updates, givens={
                x: data[i * batch_size: (i + 1) * batch_size]
            })

            start_time = time.clock()
            for mini_batch_i in xrange(mini_batches):
                fine_tune(mini_batch_i)
            end_time = time.clock()

        print ('... fine tuning took %f minutes' % ((end_time - start_time) / 60))

