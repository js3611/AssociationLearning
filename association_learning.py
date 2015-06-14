import theano
import theano.tensor as T
import numpy as np
from rbm import RBM
from rbm_config import *
from rbm_units import *
from rbm_logger import *
import DBN
import associative_dbn
import utils
import m_loader as m_loader
import datastorage as store
# from matplotlib.pyplot import plot, show, ion
# import matplotlib.pyplot as plt
from simple_classifiers import SimpleClassifier

import logging

logging.basicConfig(filename='trace.log', level=logging.INFO)


def associate_data2label(cache=False):
    cache=True
    print "Testing ClassRBM with generative target (i.e. AssociativeRBM with picture-label association)"

    # Load mnist hand digits, class label is already set to binary
    train, valid, test = m_loader.load_digits(n=[50000, 100, 30], pre={'label_vector': True})
    train_x, train_y = train
    test_x, test_y = test
    train_y = T.cast(train_y, dtype=theano.config.floatX)
    test_y = T.cast(test_y, dtype=theano.config.floatX)

    # Initialise the RBM and training parameters
    tr = TrainParam(learning_rate=0.001,
                    momentum_type=CLASSICAL,
                    momentum=0.5,
                    weight_decay=0.0001,
                    sparsity_constraint=False,
                    sparsity_target=0.1,
                    sparsity_cost=0.01,
                    sparsity_decay=0.9,
                    dropout=True,
                    dropout_rate=0.8,
                    epochs=20)

    n_visible = train_x.get_value().shape[1]
    n_visible2 = 10
    n_hidden = 500

    config = RBMConfig(v_n=n_visible,
                       v2_n=n_visible2,
                       h_n=n_hidden,
                       associative=True,
                       cd_type=CLASSICAL,
                       cd_steps=1,
                       train_params=tr,
                       progress_logger=ProgressLogger())
    rbm = RBM(config)

    store.move_to('label_test/' + str(rbm))

    loaded = store.retrieve_object(str(rbm))
    if loaded and cache:
        rbm = loaded
        print "... loaded precomputed rbm"
    else:
        rbm.train(train_x, train_y)
        rbm.save()


    # rbm.associative = False
    # rbm.reconstruct(test_x, 10)
    # rbm.associative = True

    y = map(np.argmax, test_y.eval())
    pred = rbm.classify(test_x)
    print y
    print pred
    score = np.sum(y == pred) * 1. / len(y)

    print "Classification Rate: {}".format(score)


def associate_data2data(cache=False, train_further=True):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('EvenOddP', log=True)
    train_n = 10000
    test_n = 1000
    # Load mnist hand digits, class label is already set to binary
    dataset = m_loader.load_digits(n=[train_n, 100, test_n],
                                   digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   pre={'binary_label': True})

    tr_x, tr_y = dataset[0]
    te_x, te_y = dataset[2]
    tr_x01 = m_loader.sample_image(tr_y)
    te_x01 = m_loader.sample_image(te_y)
    ones = m_loader.load_digits(n=[test_n, 0, 0], digits=[1])[0][0]
    zeroes = m_loader.load_digits(n=[test_n, 0, 0], digits=[0])[0][0]

    concat1 = theano.function([], T.concatenate([tr_x, tr_x01], axis=1))()
    # concat2 = theano.function([], T.concatenate([tr_x01, tr_x], axis=1))()
    # c = np.concatenate([concat1, concat2], axis=0)
    # np.random.shuffle(c)
    # tr_concat_x = theano.shared(c, name='tr_concat_x')
    tr_concat_x = theano.shared(concat1, name='tr_concat_x')

    # Initialise the RBM and training parameters
    # tr = TrainParam(learning_rate=0.1,
    # momentum_type=NESTEROV,
    # momentum=0.5,
    # weight_decay=0.001,
    # sparsity_constraint=True,
    # sparsity_target=0.1 ** 9,
    # sparsity_cost=0.9,
    # sparsity_decay=0.99,
    #                 epochs=50)

    tr = TrainParam(learning_rate=0.001,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.1,
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_decay=0.9,
                    sparsity_cost=0.1,
                    dropout=True,
                    dropout_rate=0.5,
                    epochs=1)

    # Even odd test
    k = 1
    n_visible = 784 * 2
    n_visible2 = 0

    # Hinton way
    # 10 classes that are equi-probable: p(x) = 0.1
    # n_hidden = min(1000, int((- np.log2(0.1)) * train_n / 10))
    n_hidden = 1000
    print "number of hidden nodes: %d" % n_hidden

    config = RBMConfig(v_n=n_visible,
                       v2_n=n_visible2,
                       h_n=n_hidden,
                       cd_type=CLASSICAL,
                       cd_steps=k,
                       train_params=tr,
                       progress_logger=ProgressLogger(img_shape=(28 * 2, 28)))

    rbm = RBM(config=config)

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded and cache:
        rbm = loaded
        print "... loaded precomputed rbm"

    errors = []
    for i in xrange(0, 10):
        # Train RBM
        if not loaded or train_further:
            rbm.train(tr_concat_x)

        # Save RBM
        data_manager.persist(rbm)

        # Reconstruct using RBM
        y = theano.shared(rbm.np_rand.binomial(1, 0, size=(test_n, 784)).astype(t_float_x))

        recon_x = rbm.mean_field_inference_opt(te_x,
                                               y,
                                               # te_x01,
                                               sample=False,
                                               k=5,
                                               img_name="te_recon_%d" % i)


        # Compare free energy
        te_x_one = theano.function([], T.concatenate([te_x, ones], axis=1))()
        te_x_one = theano.shared(te_x_one, name='te_x_one')
        te_x_zero = theano.function([], T.concatenate([te_x, zeroes], axis=1))()
        te_x_zero = theano.shared(te_x_zero, name='te_x_zero')

        e0 = rbm.free_energy(te_x_zero)
        e1 = rbm.free_energy(te_x_one)

        e0, e1 = theano.function([], [e0, e1])()
        # take column that's bigger

        # print e0
        # print e1
        #
        # print e1 > e0

        clf_tr = rbm.mean_field_inference_opt(te_x,
                                              te_x01,
                                              sample=False,
                                              k=10,
                                              img_name="tr_recon_%d" % i)

        clf = SimpleClassifier('logistic', te_x.get_value(), te_y.eval())
        orig = te_y.eval()
        error = clf.get_score(recon_x, orig)
        print error
        errors.append(error)

    # plt.plot(errors)
    # plt.show()
    print errors


def get_p_h(brain_c, tr_x, tr_x01):
    rbm_top = brain_c.association_layer
    dbn_left = brain_c.dbn_left
    dbn_right = brain_c.dbn_right
    left_in = dbn_left.bottom_up_pass(tr_x)
    right_in = dbn_right.bottom_up_pass(tr_x01)
    _, p_h = rbm_top.prop_up(np.concatenate((left_in, right_in), axis=1))
    p_h = theano.function([], T.mean(p_h, axis=0))()
    return p_h


# def plot_hidden_activity(brain_c, tr_x, tr_x01):
# p_h = get_p_h(brain_c, tr_x.get_value(), tr_x01.get_value())
# plt.clf()
#     plt.figure(0)
#     plt.plot(p_h)
#     # plt.show(block=False)
#
#     t0 = m_loader.load_digits(shared=False, n=[100, 0, 0], digits=[0])[0][0]
#     t1 = m_loader.load_digits(shared=False, n=[100, 0, 0], digits=[1])[0][0]
#     zero = np.zeros((100, 784)).astype(t_float_x)
#
#     plt.figure(1)
#     plt.subplot(321)
#     p_h1 = get_p_h(brain_c, t0, t0)
#     plt.plot(p_h1)
#     plt.subplot(322)
#     p_h2 = get_p_h(brain_c, t0, zero)
#     plt.plot(p_h2)
#     plt.show(block=False)


def associate_data2dataADBN(cache=False, train_further=True):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    f = open('adbnlog.txt', 'a')
    # project set-up
    data_manager = store.StorageManager('AssDBN_digits_Long', log=False)
    shape = 28
    train_n = 10000
    test_n = 1000
    # Load mnist hand digits, class label is already set to binary
    dataset = m_loader.load_digits(n=[train_n, 100, test_n],
                                   digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   pre={'binary_label': True})

    tr_x, tr_y = dataset[0]
    te_x, te_y = dataset[2]
    tr_x01 = m_loader.sample_image(tr_y)
    te_x01 = m_loader.sample_image(te_y)
    ones = m_loader.load_digits(n=[test_n, 0, 0], digits=[1])[0][0]
    zeroes = m_loader.load_digits(n=[test_n, 0, 0], digits=[0])[0][0]

    # brain_c = get_brain_model_AssociativeDBN(shape, data_manager)
    # brains = get_adbns(data_manager)
    # brain_c.train(tr_x, tr_x01,
    #               cache=[[True, True, True], [True, True, True], True],
    #               train_further=[[True, True, True], [True, True, True], False])

    # brain_c.train(tr_x, tr_x01,
    #               cache=[[True, True, True], [True, True, True], True],
    #               train_further=[[False, True, True], [False, True, True], False])
    #
    clf = SimpleClassifier('logistic', te_x.get_value(), te_y.eval())
    #
    #
    # for i in [3, 5, 10]:
    #     recon_x = brain_c.recall(te_x, associate_steps=i, recall_steps=0, img_name='adbn_child_recon_{}1'.format(shape), y_type='sample_active_h')
    #     error = clf.get_score(recon_x, te_y.eval())
    #     print "sample_active_h %f" % error
    #
    #     recon_x = brain_c.recall(te_x, associate_steps=i, recall_steps=0, img_name='adbn_child_recon_{}2'.format(shape), y_type='active_h')
    #     error = clf.get_score(recon_x, te_y.eval())
    #     print "active_h %f" % error
    #
    #     recon_x = brain_c.recall(te_x, associate_steps=i, recall_steps=0, img_name='adbn_child_recon_{}3'.format(shape), y_type='noisy_active_h')
    #     error = clf.get_score(recon_x, te_y.eval())
    #     print "noisy_active_h %f" % error
    #
    #     recon_x = brain_c.recall(te_x, associate_steps=i, recall_steps=0, img_name='adbn_child_recon_{}4'.format(shape), y_type='v_noisy_active_h')
    #     error = clf.get_score(recon_x, te_y.eval())
    #     print "v_noisy_active_h %f" % error
    #
    #
    #
    # print error

    for dropout in [True, False]:
        for sc in [True, False]:
            for lr in [0.001, 0.0001, 0.00005, 0.00001]:
                for n in [100, 250, 500]:
                    config = get_brain_model_AssociativeDBNConfig(28, data_manager=data_manager)
                    config.top_rbm.train_params.learning_rate = lr
                    config.top_rbm.train_params.sparsity_constraint = sc
                    config.top_rbm.train_params.dropout = dropout
                    config.n_association = n
                    adbn = associative_dbn.AssociativeDBN(config=config, data_manager=data_manager)
                    brain_c = adbn
                    f.write(str(brain_c.association_layer) + "\n")

                    errors = []
                    for i in xrange(0, 5):
                        f.write("Epoch %d \n" % (i * 10))

                        brain_c.train(tr_x, tr_x01,
                                      cache=[[True, True, True], [True, True, True], False],
                                      train_further=[[False, True, True], [False, True, True], True])

                        if i == 0:
                            # Reconstruction
                            recon_right = brain_c.dbn_left.reconstruct(tr_x, k=10, plot_every=1, plot_n=100,
                                                                       img_name='adbn_left_recon_{}'.format(shape))
                            recon_left = brain_c.dbn_right.reconstruct(tr_x01, k=10, plot_every=1, plot_n=100,
                                                                       img_name='adbn_right_recon_{}'.format(shape))

                        # Plot hidden activity
                        # plot_hidden_activity(brain_c, tr_x, tr_x01)

                        # recon_x = brain_c.recall(te_x, associate_steps=5, recall_steps=0, img_name='adbn_child_recon_{}'.format(shape))
                        #
                        # clf = SimpleClassifier('logistic', te_x.get_value(), te_y.eval())

                        for j in [5, 10]:
                            recon_x = brain_c.recall(te_x, associate_steps=j, recall_steps=0,
                                                     img_name='adbn_child_recon_{}1'.format(shape), y_type='active_h')
                            error = clf.get_score(recon_x, te_y.eval())
                            f.write("active_h %f\n" % error)

                            recon_x = brain_c.recall(te_x, associate_steps=j, recall_steps=0,
                                                     img_name='adbn_child_recon_{}2'.format(shape),
                                                     y_type='v_noisy_active_h')
                            error = clf.get_score(recon_x, te_y.eval())
                            f.write("v_noisy_active_h %f\n" % error)

                            # error = clf.get_score(recon_x, te_y.eval())
                            # print error
                            # errors.append(error)

                            # plt.plot(errors)
                            # plt.show()
                            # print errors
    # plt.show(block=True)
    f.close()


def get_brain_model_AssociativeDBNConfig(shape):
    # initialise AssociativeDBN
    config = associative_dbn.DefaultADBNConfig()

    # Gaussian Input Layer
    bottom_tr = TrainParam(learning_rate=0.001,
                           momentum_type=NESTEROV,
                           momentum=0.9,
                           weight_decay=0.0001,
                           sparsity_constraint=True,
                           sparsity_target=0.1,
                           sparsity_decay=0.9,
                           sparsity_cost=0.1,
                           dropout=True,
                           dropout_rate=0.5,
                           epochs=10)

    bottom_tr_r = TrainParam(learning_rate=0.001,
                             momentum_type=NESTEROV,
                             momentum=0.5,
                             weight_decay=0.0001,
                             sparsity_constraint=True,
                             sparsity_target=0.1,
                             sparsity_decay=0.9,
                             sparsity_cost=0.1,
                             dropout=True,
                             dropout_rate=0.5,
                             epochs=10)

    rest_tr = TrainParam(learning_rate=0.0001,
                         momentum_type=NESTEROV,
                         momentum=0.5,
                         weight_decay=0.0001,
                         dropout=True,
                         dropout_rate=0.8,
                         epochs=10,
                         batch_size=20)

    h_n = 300
    h_n_r = 50
    bottom_logger = ProgressLogger(img_shape=(shape, shape))
    bottom_rbm = RBMConfig(v_n=shape ** 2,
                           h_n=h_n,
                           progress_logger=bottom_logger,
                           train_params=bottom_tr)

    bottom_rbm_r = RBMConfig(v_n=shape ** 2,
                             h_n=h_n_r,
                             progress_logger=bottom_logger,
                             train_params=bottom_tr_r)

    rest_logger = ProgressLogger()
    rest_rbm = RBMConfig(v_n=250,
                         h_n=250,
                         progress_logger=rest_logger,
                         train_params=rest_tr)

    config.left_dbn.rbm_configs = [bottom_rbm]  # , rest_rbm]
    config.right_dbn.rbm_configs = [bottom_rbm_r]  # , rest_rbm]
    config.left_dbn.topology = [shape ** 2, h_n]  # , 250]
    config.right_dbn.topology = [shape ** 2, h_n_r]  # , 250]

    top_tr = TrainParam(learning_rate=0.00001,
                        momentum_type=NESTEROV,
                        momentum=0.9,
                        weight_decay=0.0001,
                        sparsity_constraint=False,
                        sparsity_target=0.1,
                        sparsity_decay=0.9,
                        sparsity_cost=0.1,
                        dropout=True,
                        dropout_rate=0.5,
                        batch_size=10,
                        epochs=10
                        )

    config.top_rbm.train_params = top_tr
    config.n_association = 300
    config.reuse_dbn = False
    return config


def get_brain_model_JointDBNConfig(shape, data_manager):
    config = DBN.DBNConfig(data_manager=data_manager)

    # Layer 1
    bottom_tr = TrainParam(learning_rate=0.001,
                           momentum_type=NESTEROV,
                           momentum=0.5,
                           weight_decay=0.00001,
                           sparsity_constraint=True,
                           sparsity_target=0.1,
                           sparsity_decay=0.9,
                           sparsity_cost=0.1,
                           dropout=True,
                           dropout_rate=0.5,
                           epochs=2)

    h_n = 300
    bottom_logger = ProgressLogger(img_shape=(shape * 2, shape))
    bottom_rbm = RBMConfig(v_n=(shape ** 2) * 2,
                           h_n=h_n,
                           progress_logger=bottom_logger,
                           train_params=bottom_tr)

    # Layer 2
    rest_tr = TrainParam(learning_rate=0.0001,
                         momentum_type=NESTEROV,
                         momentum=0.5,
                         weight_decay=0.0001,
                         dropout=True,
                         dropout_rate=0.8,
                         epochs=10,
                         batch_size=20)

    rest_logger = ProgressLogger()
    rest_rbm = RBMConfig(v_n=250,
                         h_n=250,
                         progress_logger=rest_logger,
                         train_params=rest_tr)

    # Layer 3
    top_tr = TrainParam(learning_rate=0.0001,
                        momentum_type=NESTEROV,
                        momentum=0.0,
                        weight_decay=0.0000,
                        dropout=False,
                        dropout_rate=0.8,
                        epochs=10,
                        batch_size=5)

    top_h_n = 100
    top_rbm = RBMConfig(v_n=h_n,
                        h_n=top_h_n,
                        progress_logger=rest_logger,
                        train_params=rest_tr)

    config.rbm_configs = [bottom_rbm, top_rbm]  # , rest_rbm]
    config.topology = [(shape ** 2) * 2, h_n, top_h_n]  # , 250]
    return config


def get_adbns():
    adbns_configs = []

    # for dropout in [True, False]:
    #     for sc in [True, False]:
    #         for lr in [0.001, 0.0001, 0.00005, 0.00001]:

    for n in [100, 250, 500, 1000]:
        for left_n in [100, 250, 500]:
            config = get_brain_model_AssociativeDBNConfig(28)
            config.left_dbn.topology[1] = left_n
            config.left_dbn.rbm_configs[0].h_n = left_n
            # config.top_rbm.train_params.learning_rate=lr
            # config.top_rbm.train_params.sparsity_constraint=sc
            # config.top_rbm.train_params.dropout=dropout
            config.n_association = n
            adbns_configs.append(config)

    return adbns_configs


def associate_data2dataDBN(cache=False):
    print "Testing Associative DBN which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('associative_dbn_test', log=True)


    # Load mnist hand digits, class label is already set to binary
    train, valid, test = m_loader.load_digits(n=[500, 100, 100], digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = m_loader.sample_image(train_y)

    dataset01 = m_loader.load_digits(n=[500, 100, 100], digits=[0, 1])

    # Initialise RBM parameters
    # fixed base train param
    base_tr = RBM.TrainParam(learning_rate=0.01,
                             momentum_type=RBM.CLASSICAL,
                             momentum=0.5,
                             weight_decay=0.0005,
                             sparsity_constraint=False,
                             epochs=20)

    # top layer parameters
    tr = RBM.TrainParam(learning_rate=0.1,
                        find_learning_rate=True,
                        momentum_type=RBM.NESTEROV,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=False,
                        epochs=20)

    tr_top = RBM.TrainParam(learning_rate=0.1,
                            find_learning_rate=True,
                            momentum_type=RBM.CLASSICAL,
                            momentum=0.5,
                            weight_decay=0.001,
                            sparsity_constraint=False,
                            epochs=20)


    # Layer 1
    # Layer 2
    # Layer 3
    topology = [784, 500, 500, 100]

    config = associative_dbn.DefaultADBNConfig()
    config.topology_left = [784, 500, 500, 100]
    config.topology_right = [784, 500, 500, 100]
    config.reuse_dbn = False
    config.top_rbm_params = tr_top
    config.base_rbm_params = [base_tr, tr, tr]

    for cd_type in [RBM.CLASSICAL, RBM.PERSISTENT]:
        for n_ass in [100, 250, 500, 750, 1000]:
            config.n_association = n_ass
            config.top_cd_type = cd_type

            # Construct DBN
            ass_dbn = associative_dbn.AssociativeDBN(config=config, data_manager=data_manager)

            # Train
            ass_dbn.train(train_x, train_x01, cache=cache, optimise=True)

            for n_recall in [1, 3, 5, 7, 10]:
                for n_think in [0, 1, 3, 5, 7, 10]:  # 1, 3, 5, 7, 10]:
                    # Reconstruct
                    sampled = ass_dbn.recall(test_x, n_recall, n_think)

                    # Sample from top layer to generate data
                    sample_n = 100
                    utils.save_images(sampled, image_name='reconstruced_{}_{}_{}.png'.format(n_ass, n_recall, n_think),
                                      shape=(sample_n / 10, 10))

                    dataset01[2] = (theano.shared(sampled), test_y)

                    # Classify the reconstructions TODO
                    # score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset01, 100)
                    #
                    # print 'Score: {}'.format(str(score))
                    # logging.info('{}, {}, {}, {}: {}'.format(cd_type, n_ass, n_recall, n_think, score))


def associate_data2dataJDBN(cache=False, train_further=False):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    f = open('errors.txt', 'w')
    # project set-up
    proj_name = 'JDBN_digits'
    data_manager = store.StorageManager(proj_name, log=False)
    shape = 28
    train_n = 10000
    test_n = 1000
    # Load mnist hand digits, class label is already set to binary
    dataset = m_loader.load_digits(n=[train_n, 0, test_n],
                                   digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   pre={'binary_label': True})

    tr_x, tr_y = dataset[0]
    te_x, te_y = dataset[2]
    tr_x01 = m_loader.sample_image(tr_y)
    te_x01 = m_loader.sample_image(te_y)
    ones = m_loader.load_digits(n=[test_n, 0, 0], digits=[1])[0][0]
    zeroes = m_loader.load_digits(n=[test_n, 0, 0], digits=[0])[0][0]
    tr_X = theano.shared(np.concatenate([tr_x.get_value(), tr_x01.get_value()], axis=1))
    initial_y = np.random.binomial(n=1, p=0.0, size=te_x.get_value(True).shape).astype(t_float_x)
    initial_y_uni = np.random.uniform(low=0, high=1, size=te_x.get_value(True).shape).astype(t_float_x)
    initial_y_bi001 = np.random.binomial(n=1, p=0.01, size=te_x.get_value(True).shape).astype(t_float_x)
    initial_y_bi01 = np.random.binomial(n=1, p=0.1, size=te_x.get_value(True).shape).astype(t_float_x)
    te_X = theano.shared(np.concatenate([te_x.get_value(), initial_y], axis=1))
    te_X2 = theano.shared(np.concatenate([te_x.get_value(), initial_y_uni], axis=1))
    te_X3 = theano.shared(np.concatenate([te_x.get_value(), initial_y_bi01], axis=1))
    te_X4 = theano.shared(np.concatenate([te_x.get_value(), initial_y_bi001], axis=1))
    clf = SimpleClassifier('logistic', te_x01.get_value(), te_y.eval())

    configs = []

    for h_n in [100, 250, 500]:
        config1 = get_brain_model_JointDBNConfig(shape, data_manager)
        config1.topology = [784 * 2, h_n, h_n]
        config1.rbm_configs[0].h_n = h_n
        config1.rbm_configs[1].v_n = h_n
        config1.rbm_configs[1].h_n = h_n
        configs.append(config1)

    for h_n in [100, 250, 500]:
        config1 = get_brain_model_JointDBNConfig(shape, data_manager)
        config1.topology = [784 * 2, h_n, h_n * 2]
        config1.rbm_configs[0].h_n = h_n
        config1.rbm_configs[1].v_n = h_n
        config1.rbm_configs[1].h_n = h_n * 2
        configs.append(config1)

    for h_n in [100, 250, 500]:
        config1 = get_brain_model_JointDBNConfig(shape, data_manager)
        config1.topology = [784 * 2, h_n * 2, h_n]
        config1.rbm_configs[0].h_n = h_n * 2
        config1.rbm_configs[1].v_n = h_n * 2
        config1.rbm_configs[1].h_n = h_n
        configs.append(config1)

    for a in xrange(10):
        for config in configs:
            f.write('{}:{}:'.format(a, config.topology))
            brain_c = DBN.DBN(config=config)
            brain_c.pretrain(tr_X, cache=[True, True], train_further=[True, True])

            recon_x = brain_c.reconstruct(te_X, k=1, plot_n=100, img_name='{}_{}_recon'.format(a, config.topology))
            recon = recon_x[:, 784:]
            error = clf.get_score(recon, te_y.eval())
            print error
            f.write('{}, '.format(error))

            for i in xrange(0, 5):
                brain_c.fine_tune(tr_X)
                recon_x = brain_c.reconstruct(te_X, k=1, plot_n=100,
                                              img_name=('{}_{}_recon_fine_tune{}'.format(a, config.topology, i)))
                recon = recon_x[:, 784:]
                error = clf.get_score(recon, te_y.eval())
                print error
                f.write('{}, '.format(error))
                recon_x = brain_c.reconstruct(te_X4, k=1, plot_n=100,
                                              img_name=('{}_{}_recon_fine_tune_2_{}'.format(a, config.topology, i)))
                recon = recon_x[:, 784:]
                error = clf.get_score(recon, te_y.eval())
                print error
                f.write('{}, '.format(error))
            f.write('\n')
    f.close()


def associate_data2dataADBN_Finetune(cache=False, train_further=False):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    f = open('adbn_errors.txt', 'w')
    # project set-up
    proj_name = 'ADBN_digits'
    data_manager = store.StorageManager(proj_name, log=True)
    shape = 28
    train_n = 10000
    test_n = 1000
    # Load mnist hand digits, class label is already set to binary
    dataset = m_loader.load_digits(n=[train_n, 0, test_n],
                                   digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   pre={'binary_label': True})

    tr_x, tr_y = dataset[0]
    te_x, te_y = dataset[2]
    tr_x01 = m_loader.sample_image(tr_y)
    te_x01 = m_loader.sample_image(te_y)
    clf = SimpleClassifier('logistic', te_x01.get_value(), te_y.eval())

    configs = get_adbns()

    for a in xrange(10):
        for n, config in enumerate(configs):
            t = 'l{}_r{}_t{}'.format(config.left_dbn.topology, config.right_dbn.topology, config.n_association)
            f.write('{}:{}:'.format(a, t))
            brain_c = associative_dbn.AssociativeDBN(config=config,
                                                     data_manager=store.StorageManager('{}/{}'.format(proj_name, n),
                                                                                       log=False))
            brain_c.train(tr_x, tr_x01, cache=True, train_further=True)

            recon = brain_c.recall(tr_x,
                                   associate_steps=10,
                                   recall_steps=0,
                                   img_name='{}_{}'.format(a, t))

            error = clf.get_score(recon, tr_y.eval())
            print error
            f.write('{}, '.format(error))

            for i in xrange(0, 10):
                brain_c.fine_tune(tr_x, tr_x01, epochs=1)
                for y_type in ['active_h', 'v_noisy_active_h', 'zero']:
                    recon = brain_c.recall(tr_x,
                                           associate_steps=10,
                                           recall_steps=0,
                                           img_name='{}_{}_{}_{}'.format(a, t, y_type, i),
                                           y_type=y_type)
                    error = clf.get_score(recon, tr_y.eval())
                    print error
                    f.write('{}, '.format(error))
            f.write('\n')
    f.close()


if __name__ == '__main__':
    # associate_data2label()
    associate_data2data(True, True)
    # associate_data2dataADBN(True, True)
    # associate_data2dataADBN_Finetune(True, True)
    # associate_data2dataJDBN(True, False)
    # associate_data2dataDBN(False)

