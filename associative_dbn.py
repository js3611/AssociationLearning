from rbm import *
from DBN import *
from simple_classifiers import SimpleClassifier

try:
    import PIL.Image as Image
except ImportError:
    import Image

from utils import save_images

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class DefaultADBNConfig(object):
    def __init__(self):
        # Base RBM Parameters
        # Layer 1
        tr = TrainParam(learning_rate=0.001,
                        momentum_type=NESTEROV,
                        momentum=0.5,
                        weight_decay=0.0001,
                        epochs=20)

        first_progress_logger = ProgressLogger(img_shape=(25, 25))
        first_rbm_config = RBMConfig(train_params=tr,
                                     progress_logger=first_progress_logger)

        # Layer 2 onwards
        tr2 = tr
        rest_progress_logger = ProgressLogger()
        rbm_config = RBMConfig(train_params=tr,
                               progress_logger=rest_progress_logger)

        # Left DBN
        left_topology = [625, 100]
        tr_list = [tr, tr2, tr2]
        rbm_configs = [first_rbm_config, rbm_config, rbm_config]
        left_dbn_config = DBNConfig(out_dir='left',
                                    topology=left_topology,
                                    training_parameters=tr_list,
                                    rbm_configs=rbm_configs)

        # Right DBN
        right_topology = [625, 100]
        right_dbn_config = DBNConfig(out_dir='right',
                                     topology=right_topology,
                                     training_parameters=tr_list,
                                     rbm_configs=rbm_configs)

        self.opt_top = True

        if self.opt_top:
            top_rbm_config = RBMConfig(train_params=tr, progress_logger=ProgressLogger())
            top_rbm_config.associative = False
        else:
            # Top AssociativeRBM
            top_rbm_config = RBMConfig(train_params=tr, progress_logger=AssociationProgressLogger())
            top_rbm_config.associative = True

        # Set configurations
        self.reuse_dbn = False
        self.left_dbn = left_dbn_config
        self.right_dbn = right_dbn_config
        self.top_rbm = top_rbm_config
        self.n_association = 100


class AssociativeDBN(object):
    def __init__(self, config, data_manager=None):

        # Set parameters / Assertion
        config.left_dbn.data_manager = data_manager
        config.right_dbn.data_manager = data_manager
        top_rbm_config = config.top_rbm
        top_rbm_config.h_n = config.n_association
        v_n = config.left_dbn.topology[-1]
        v2_n = config.right_dbn.topology[-1]
        if config.opt_top:
            top_rbm_config.v_n = v_n + v2_n
        else:
            top_rbm_config.v_n = config.left_dbn.topology[-1]
            top_rbm_config.v2_n = config.right_dbn.topology[-1]

        config.top_rbm = top_rbm_config
        self.config = config
        self.opt_top = config.opt_top
        self.data_manager = data_manager
        self.dbn_left = DBN(config.left_dbn)
        self.dbn_right = DBN(config.right_dbn) if not config.reuse_dbn else self.dbn_left
        print '... initialising association layer'
        self.association_layer = RBM(config=config.top_rbm)

    def __str__(self):
        return 'l{}_r{}_t{}'.format(self.dbn_left, self.dbn_right, self.association_layer.h_n)

    def train(self, x1, x2, cache=False, train_further=False):
        cache_left = cache_right = cache_top = cache if type(cache) is bool else False
        train_further_left = train_further_right = train_further_top = train_further if type(
            train_further) is bool else False
        if type(cache) is list:
            cache_left = cache[0]
            cache_right = cache[1]
            cache_top = cache[2]
        if type(train_further) is list:
            train_further_left = train_further[0]
            train_further_right = train_further[1]
            train_further_top = train_further[2]

        # Train left & right DBN's
        self.dbn_left.pretrain(x1, cache=cache_left,
                               train_further=train_further_left)

        if self.config.reuse_dbn:
            self.dbn_right = self.dbn_left
        else:
            self.dbn_right.pretrain(x2, cache=cache_right,
                                    train_further=train_further_right)

        # Pass the parameter to top layer
        x1_np = self.dbn_left.bottom_up_pass(x1.get_value(True))
        x2_np = self.dbn_right.bottom_up_pass(x2.get_value(True))
        x1_features = theano.shared(x1_np)
        x2_features = theano.shared(x2_np)

        # Train top association layer

        top = self.association_layer
        tr = top.train_parameters

        # Check Cache
        out_dir = 'association_layer/{}_{}/'.format(len(self.dbn_left.rbm_layers),
                                                    len(self.dbn_right.rbm_layers))

        load = self.data_manager.retrieve('{}_{}'.format(self.opt_top, top),
                                          out_dir=out_dir)

        if load and cache_top:
            self.association_layer = load
            print '... top layer RBM loaded'

        if not load and tr.sparsity_constraint:
            top.set_initial_hidden_bias()
            if self.opt_top:
                # Concatenate images
                x = theano.shared(np.concatenate((x1_np, x2_np), axis=1))
                top.set_hidden_mean_activity(x)
            else:
                top.set_hidden_mean_activity(x1_features, x2_features)

        if not load or train_further_top:
            if self.opt_top:
                # Concatenate images
                x = theano.shared(np.concatenate((x1_np, x2_np), axis=1))
                top.train(x)
            else:
                top.train(x1_features, x2_features)

            self.data_manager.persist(top,
                                      '{}_{}'.format(self.opt_top, top),
                                      out_dir=out_dir)

    def recall(self, x, associate_steps=10, recall_steps=5, img_name='dbn', y=None, y_type='sample_active_h'):
        ''' left dbn bottom-up -> associate -> right dbn top-down
        :param x: data
        :param associate_steps: top level gibbs sampling steps
        :param recall_steps: right dbn sampling
        :return:
        '''
        self.data_manager.move_to('reconstruct')
        if self.data_manager.log:
            print '... moved to {}'.format(os.getcwd())

        left = self.dbn_left
        top = self.association_layer
        right = self.dbn_right

        if utils.isSharedType(x):
            x = x.get_value(borrow=True)

        # Pass to association layer
        top_out = left.bottom_up_pass(x)
        assoc_in = theano.shared(top_out, 'top_in', allow_downcast=True)

        # Sample from the association layer
        # associate_x = top.reconstruct_association(assoc_in, k=associate_steps)
        if self.opt_top:
            # Initialise y according to the neuron distribution
            if type(top.v_unit) is GaussianVisibleUnit:
                # TODO
                print 'GAUSSIAN INPUT IS NOT SUPPORTED'

            top_shape = top_out.shape[0]
            right_top_rbm = right.rbm_layers[-1]
            shape = (top_shape, right_top_rbm.h_n)
            y_base = np.zeros(shape).astype(t_float_x)
            p = right_top_rbm.active_probability_h.get_value(borrow=True)

            if y_type == 'sample_active_h' or type(y) is None:
                print 'initialise reconstruction by active_h'

                # import matplotlib.pyplot as plt
                # plt.plot(p)
                # plt.show()

                y_base = right_top_rbm.np_rand.binomial(size=shape,
                                                        n=1,
                                                        p=p).astype(t_float_x)

            if y_type == 'active_h':
                y_base = np.tile(p, (shape[0], 1)).astype(t_float_x)

            if y_type == 'v_noisy_active_h':
                y_base = right_top_rbm.np_rand.normal(loc=0, scale=0.2, size=shape) + np.tile(p, (shape[0], 1))
                y_base = y_base.astype(t_float_x)

            if y_type == 'noisy_active_h':
                y_base = right_top_rbm.np_rand.normal(loc=0, scale=0.1, size=shape) + np.tile(p, (shape[0], 1))
                y_base = y_base.astype(t_float_x)

            if 'binomial' in y_type:
                p = float(y_type.strip('binomial'))
                y_base = right_top_rbm.np_rand.binomial(size=shape,
                                                        n=1,
                                                        p=p).astype(t_float_x)

            y = theano.shared(y_base, name='assoc_y')
            associate_x = top.mean_field_inference_opt(assoc_in, y=y, sample=True, k=associate_steps)
        else:
            associate_x = top.mean_field_inference(assoc_in, sample=True, k=associate_steps)
        # associate_x = top.reconstruct_association(assoc_in, k=associate_steps)

        if recall_steps > 0:
            top_in = theano.shared(associate_x, 'associate_x', allow_downcast=True)
            # Allow right dbn to day dream by extracting top layer rbm
            right_top_rbm = right.rbm_layers[-1]
            ass, ass_p, ass_s = right_top_rbm.sample_v_given_h(top_in)
            associate_x_in = theano.function([], ass_s)()
            associate_x_reconstruct = right_top_rbm.reconstruct(associate_x_in,
                                                                k=recall_steps,
                                                                img_name='recall')

            # pass down to visible units, take the penultimate layer because we sampled at the top layer
            if len(right.rbm_layers) > 1:
                res = right.top_down_pass(associate_x_reconstruct, start=len(right.rbm_layers) - 1)
            else:
                res = associate_x_reconstruct
                # res = result.get_value(borrow=True)
        else:
            res = right.top_down_pass(associate_x.astype(t_float_x))

        n = res.shape[0]

        img_shape = right.rbm_layers[0].track_progress.img_shape
        save_images(x, img_name + '_orig.png', shape=(n / 10, 10), img_shape=img_shape)
        save_images(res, img_name + '_recon.png', shape=(n / 10, 10), img_shape=img_shape)

        self.data_manager.move_to_project_root()

        return res

    def fine_tune_cd(self, wake_state):
        # CONTRASTIVE DIVERGENCE AT TOP LAYER
        rbm = self.association_layer
        pen_state = wake_state
        [updates, chain_end, _, _, _, _, _, _, _] = rbm.negative_statistics(pen_state)
        cost = T.mean(rbm.free_energy(pen_state)) - T.mean(rbm.free_energy(chain_end))
        grads = T.grad(cost, rbm.params, consider_constant=[chain_end])
        lr = rbm.train_parameters.learning_rate
        for (p, g) in zip(rbm.params, grads):
            # TODO all the special updates like momentum
            updates[p] = p - lr * g
        return chain_end, updates

    def get_fine_tune_updates(self, xs, ys, batch_size=10):

        # WAKE-PHASE [hid_prob, pen_prob, ...], [hid_state, pen-state,...]
        wake_probs_l, wake_states_l = self.dbn_left.wake_phase(xs)
        wake_probs_r, wake_states_r = self.dbn_right.wake_phase(ys)

        # TOP LAYER CD
        wake_state = T.concatenate([wake_states_l[-1], wake_states_r[-1]], axis=1)
        chain_end, updates = self.fine_tune_cd(wake_state)
        left_end = self.dbn_left.topology[-1]
        chain_end_l, chain_end_r = chain_end[:, :left_end], chain_end[:, left_end:]

        # SLEEP PHASE: [hid_prob, vis_prob], [pen_state, hid_state, vis_state] ...
        sleep_probs_l, sleep_states_l = self.dbn_left.sleep_phase(chain_end_l)
        sleep_probs_r, sleep_states_r = self.dbn_right.sleep_phase(chain_end_r)

        # Prediction
        psleep_states_l, pwake_states_l, sleep_states_l, wake_states_l = self.dbn_left.get_predictions(xs,
                                                                                                       sleep_probs_l,
                                                                                                       sleep_states_l,
                                                                                                       wake_states_l)
        psleep_states_r, pwake_states_r, sleep_states_r, wake_states_r = self.dbn_right.get_predictions(ys,
                                                                                                        sleep_probs_r,
                                                                                                        sleep_states_r,
                                                                                                        wake_states_r)

        # UPDATES TO GENERATIVE PARAMETERS
        updates = self.dbn_left.update_generative_weights(batch_size, pwake_states_l, wake_states_l, updates)
        updates = self.dbn_right.update_generative_weights(batch_size, pwake_states_r, wake_states_r, updates)

        # UPDATES TO INFERENCE PARAMETERS
        updates = self.dbn_left.update_inference_weights(batch_size, psleep_states_l, sleep_states_l, updates)
        updates = self.dbn_right.update_inference_weights(batch_size, psleep_states_r, sleep_states_r, updates)

        return updates

    def fine_tune(self, data_r, data_l, epochs=10, batch_size=10):

        if not self.dbn_right.untied:
            self.dbn_left.untie_weights(include_top=True)
            self.dbn_right.untie_weights(include_top=True)

        mini_batches = data_r.get_value(borrow=True).shape[0] / batch_size
        i = T.iscalar()
        x = T.matrix('x')
        y = T.matrix('y')
        updates = self.get_fine_tune_updates(x, y, batch_size)
        fine_tune = theano.function([i], [], updates=updates, givens={
            x: data_r[i * batch_size: (i + 1) * batch_size],
            y: data_l[i * batch_size: (i + 1) * batch_size]
        })

        for epoch in xrange(epochs):
            print '... epoch %d' % epoch
            start_time = time.clock()
            for mini_batche_i in xrange(mini_batches):
                fine_tune(mini_batche_i)
            end_time = time.clock()

        print ('... fine tuning took %f minutes' % ((end_time - start_time) / 60))


def test_associative_dbn(i=0):
    print "Testing Associative DBN which tries to learn even-odd of numbers"

    # load dataset
    train_n = 1000
    test_n = 1000
    train, valid, test = m_loader.load_digits(n=[train_n, 100, test_n], pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = m_loader.sample_image(train_y)
    clf = SimpleClassifier('logistic', train_x01, train_y)

    # project set up
    project_name = 'AssociationDBNTest/{}'.format(i)
    data_manager = store.StorageManager(project_name, log=False)
    cache = True

    # initialise AssociativeDBN
    config = DefaultADBNConfig()
    config.reuse_dbn = False
    config.left_dbn.rbm_configs[0].progress_logger = ProgressLogger(img_shape=(28, 28))
    config.right_dbn.rbm_configs[0].progress_logger = ProgressLogger(img_shape=(28, 28))
    config.right_dbn.rbm_configs[0].train_params.epochs = 50
    config.left_dbn.topology = [784, 100]
    config.right_dbn.topology = [784, 50]
    config.n_association = 100
    associative_dbn = AssociativeDBN(config=config, data_manager=data_manager)

    # Plot sample
    save_images(train_x.get_value(borrow=True)[1:100], 'n_orig.png', (10, 10))
    save_images(train_x01.get_value(borrow=True)[1:100], 'n_ass.png', (10, 10))

    # Train RBM - learn joint distribution
    associative_dbn.train(train_x, train_x01, cache=True, train_further=False)
    print "... trained associative DBN"

    # Reconstruct images
    reconstructed_y = associative_dbn.recall(test_x,
                                             associate_steps=10,
                                             recall_steps=0,
                                             y_type='active_h')
    reconstructed_y0 = associative_dbn.recall(test_x,
                                              associate_steps=10,
                                              recall_steps=0,
                                              y_type='zero')
    print "... reconstructed images"

    # Classify the reconstructions
    score_orig = clf.get_score(reconstructed_y, test_y.eval())
    out_msg = '{} (orig, retrain):{}'.format(associative_dbn, score_orig)
    print out_msg

    # Classify the reconstructions
    score_orig = clf.get_score(reconstructed_y0, test_y.eval())
    out_msg = '{} (orig, retrain):{}'.format(associative_dbn, score_orig)
    print out_msg

    for j in xrange(1):
        # Fine tune them
        associative_dbn.fine_tune(train_x, train_x01, 1, 1)

        reconstructed_y0 = associative_dbn.recall(test_x,
                                                  associate_steps=10,
                                                  recall_steps=0,
                                                  y_type='zero',
                                                  img_name='zero_{}'.format(j))
        print "... reconstructed images"

        # Classify the reconstructions
        score_orig = clf.get_score(reconstructed_y0, test_y.eval())
        out_msg = '{} (orig, retrain):{}'.format(associative_dbn, score_orig)
        print out_msg


if __name__ == '__main__':
    test_associative_dbn()
