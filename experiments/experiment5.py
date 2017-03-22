__author__ = 'joschlemper'

import numpy as np
import theano
import theano.tensor as T
import scipy
from simple_classifiers import SimpleClassifier
from rbm import RBM
from rbm_config import *
from rbm_logger import *
from rbm_units import *
import DBN
import associative_dbn
from datastorage import StorageManager
from scipy.stats import itemfreq
import kanade_loader
import matplotlib.pyplot as plt


def experimentChild(project_name, mapping, shape, model):
    # Project set up
    data_manager = StorageManager(project_name, log=True)
    f = open(project_name + '.txt', mode='a')
    f.write(project_name)
    f.write('\tMODEL=')
    f.write(model)
    f.write('\tSHAPE=')
    f.write('%d' % shape)
    f.write('\n')

    dataset_name = 'sharp_equi{}_{}'.format(shape, shape)

    preprocesssing = {'scale': True}

    # Get dataset
    happy_set = kanade_loader.load_kanade(set_name=dataset_name,
                                          emotions=mapping.keys(),
                                          pre=preprocesssing,
                                          n=100)

    h_tr, h_vl, h_te = happy_set
    h_tr_x, h_tr_y = h_tr
    h_vl_x, h_vl_y = h_vl
    h_te_x, h_te_y = h_te

    # Sample Parent emotion
    p_tr_x, p_tr_y = kanade_loader.sample_image2(h_tr_y,
                                                 mapping=mapping,
                                                 pre=preprocesssing,
                                                 set_name=dataset_name)

    concat1 = theano.function([], T.concatenate([h_tr_x, p_tr_x], axis=1))()
    tr_x = theano.shared(concat1, name='tr_x')

    c1 = T.concatenate([h_tr_x, p_tr_x], axis=1)
    c2 = T.concatenate([p_tr_x, h_tr_x], axis=1)
    c3 = theano.function([], T.concatenate([c1, c2], axis=0))()
    tr_x_mixed = theano.shared(c3, name='tr_x_mixed')

    # TODO make interface for brain model
    # initial_y = np.zeros(h_te_x.get_value(True).shape)
    initial_y = np.random.normal(0, 1, h_te_x.get_value(True).shape)
    initial_y = theano.shared(initial_y, name='initial_y')
    te_x = theano.shared(theano.function([], T.concatenate([h_te_x, initial_y], axis=1))().astype(t_float_x))

    if model == 'rbm':
        brain_c = get_brain_model_RBM(shape)
        load = data_manager.retrieve(str(brain_c))
        if load:
            brain_c = load
        else:
            brain_c.set_initial_hidden_bias()
            brain_c.set_hidden_mean_activity(tr_x)
        brain_c.train(tr_x)
        data_manager.persist(brain_c)
        # brain_c.train(tr_x_mixed)

        recon = brain_c.reconstruct_association_opt(h_te_x, initial_y,
                                                    k=10,
                                                    plot_n=100,
                                                    img_name='rbm_child_recon_{}'.format(shape))

        recon_pair = brain_c.reconstruct(tr_x, k=1, plot_n=100, img_name='rbm_pair_recon_{}'.format(shape))
        recon_p_tr_x = recon_pair[:, (shape ** 2):]


    elif model == 'dbn':
        brain_c = get_brain_model_DBN(shape, data_manager)
        brain_c.pretrain(tr_x, cache=[True, True, True], train_further= [True, True, True])

        recon_pair = brain_c.reconstruct(tr_x, k=1, plot_n=100, img_name='dbn_pair_recon_{}'.format(shape))
        recon_p_tr_x = recon_pair[:, (shape ** 2):]

        recon_pair = brain_c.reconstruct(te_x, k=1, plot_n=100, img_name='dbn_child_recon_{}'.format(shape))
        recon = recon_pair[:, (shape ** 2):]

    elif model == 'adbn':
        brain_c = get_brain_model_AssociativeDBN(shape, data_manager)
        brain_c.train(h_tr_x, p_tr_x,
                      cache=[[True, True, True], [True, True, True], True],
                      train_further=[[True, True, True], [True, True, True], True])

        # Reconstruction
        recon_p_tr_x = brain_c.dbn_right.reconstruct(p_tr_x, k=10, plot_every=1, plot_n=100,
                                                     img_name='adbn_right_recon_{}'.format(shape))
        recon = brain_c.recall(h_te_x, associate_steps=5, recall_steps=0, img_name='adbn_child_recon_{}'.format(shape))

    # Train classifier on reconstruction
    clf = SimpleClassifier('logistic', recon_p_tr_x, p_tr_y.eval())

    # Output number of classes
    res = clf.classify(recon).T
    r = np.histogram(res, bins=np.arange(1, 9))
    labels = map(lambda x: kanade_loader.emotion_rev_dict[int(x)], r[1][:-1])
    # labels = map(lambda x: kanade_loader.emotion_rev_dict[int(x)], r[0])
    proportion = r[0] * 1. / sum(r[0])
    txt = 'Learnt Child Configuration:'
    print txt
    f.write(txt)
    f.write('\n')
    for i, l in enumerate(labels):
        fill_space = (max(map(lambda x: len(x), labels)) - len(l))
        txt = '{}{}:\t %.3f'.format(l, ' ' * fill_space) % proportion[i]
        print txt
        f.write(txt)
        f.write('\n')
    f.write('\n')
    f.close()
    data_manager.finish()


def get_brain_model_RBM(shape):
    # Initialise RBM
    tr = TrainParam(learning_rate=0.0001,
                    momentum_type=NESTEROV,
                    momentum=0.9,
                    weight_decay=0.0001,
                    sparsity_constraint=True,
                    sparsity_decay=0.9,
                    sparsity_cost=100,
                    sparsity_target=0.01,
                    batch_size=10,
                    epochs=10)
    n_visible = shape * shape * 2
    n_hidden = 500
    config = RBMConfig()
    config.v_n = n_visible
    config.h_n = n_hidden
    config.v_unit = rbm_units.GaussianVisibleUnit
    # config.h_unit = rbm_units.ReLUnit
    config.progress_logger = ProgressLogger(img_shape=(shape * 2, shape))
    config.train_params = tr
    brain_c = RBM(config)

    print "... initialised RBM"
    return brain_c


def get_brain_model_AssociativeDBN(shape, data_manager):
    # initialise AssociativeDBN
    config = associative_dbn.DefaultADBNConfig()

    # Gaussian Input Layer
    bottom_tr = TrainParam(learning_rate=0.0001,
                           momentum_type=NESTEROV,
                           momentum=0.9,
                           weight_decay=0.0001,
                           sparsity_constraint=True,
                           sparsity_decay=0.9,
                           sparsity_cost=100,
                           sparsity_target=0.01,
                           batch_size=10,
                           epochs=10)

    rest_tr = TrainParam(learning_rate=0.001,
                         momentum_type=CLASSICAL,
                         momentum=0.5,
                         weight_decay=0.0001,
                         epochs=10,
                         batch_size=10)

    h_n = 250
    bottom_logger = ProgressLogger(img_shape=(shape, shape))
    bottom_rbm = RBMConfig(v_unit=rbm_units.GaussianVisibleUnit,
                           v_n=shape ** 2,
                           h_n=h_n,
                           progress_logger=bottom_logger,
                           train_params=bottom_tr)

    config.left_dbn.rbm_configs = [bottom_rbm, rest_tr]
    config.right_dbn.rbm_configs = [bottom_rbm, rest_tr]
    config.left_dbn.topology = [shape ** 2, h_n, 250]
    config.right_dbn.topology = [shape ** 2, h_n, 250]

    top_tr = TrainParam(learning_rate=0.001,
                        momentum_type=NESTEROV,
                        momentum=0.5,
                        weight_decay=0.0001,
                        sparsity_constraint=True,
                        sparsity_target=0.01,
                        sparsity_decay=0.9,
                        sparsity_cost=1,
                        batch_size=10,
                        epochs=10
                        )

    config.top_rbm.train_params = top_tr
    config.n_association = 1000
    config.reuse_dbn = False
    adbn = associative_dbn.AssociativeDBN(config=config, data_manager=data_manager)
    print '... initialised associative DBN'
    return adbn


def get_brain_model_DBN(shape, data_manager):
    # Initialise RBM parameters
    base_tr = TrainParam(learning_rate=0.0001,
                         momentum_type=NESTEROV,
                         momentum=0.9,
                         weight_decay=0.0001,
                         sparsity_constraint=False,
                         sparsity_decay=0.9,
                         sparsity_cost=10,
                         sparsity_target=0.01,
                         batch_size=10,
                         epochs=10)

    rest_tr = TrainParam(learning_rate=0.001,
                         momentum_type=CLASSICAL,
                         momentum=0.5,
                         weight_decay=0.0001,
                         epochs=10,
                         batch_size=10)

    top_tr = TrainParam(learning_rate=0.001,
                        momentum_type=NESTEROV,
                        momentum=0.5,
                        weight_decay=0.0001,
                        sparsity_constraint=False,
                        sparsity_target=0.01,
                        sparsity_decay=0.9,
                        sparsity_cost=1,
                        batch_size=10,
                        epochs=10
                        )

    # Layer 1
    # Layer 2
    # Layer 3
    topology = [2 * (shape ** 2), 500, 500, 1000]
    # batch_size = 10
    first_progress_logger = ProgressLogger(img_shape=(shape * 2, shape))
    rest_progress_logger = ProgressLogger()

    first_rbm_config = RBMConfig(train_params=base_tr,
                                 progress_logger=first_progress_logger)
    first_rbm_config.v_unit = rbm_units.GaussianVisibleUnit
    rest_rbm_config = RBMConfig(train_params=rest_tr,
                                progress_logger=rest_progress_logger)
    top_rbm_config = RBMConfig(train_params=top_tr,
                               progress_logger=rest_progress_logger)

    rbm_configs = [first_rbm_config, rest_rbm_config, top_rbm_config]

    config = DBN.DBNConfig(topology=topology,
                           training_parameters=base_tr,
                           rbm_configs=rbm_configs,
                           data_manager=data_manager)

    # construct the Deep Belief Network
    dbn = DBN.DBN(config)
    print '... initialised DBN'
    return dbn


def plot_result(file_name, mapping, architectures=['RBN', 'DBN', 'ADBN']):
    f = open(file_name, 'r')
    emotions = set(kanade_loader.emotion_dict.keys())
    graphs = {'anger': [], 'happy': [], 'sadness': []}

    for line in f.readlines():
        splitted = ''.join(line.split()).split(':')
        if splitted[0] in graphs.keys():
            lab, val = splitted[0], splitted[1]
            graphs[lab].append(val)

    print map(len, graphs.values())
    print graphs
    f.close()

    # RBM
    anger = graphs['anger']
    happy = graphs['happy']
    sadness = graphs['sadness']
    child_reaction = mapping['sadness']
    plt.figure(1)
    attempts = np.min(map(len, graphs.values())) / len(architectures)
    print attempts

    for i in xrange(0, 3):
        plt.subplot(130 + i + 1)
        plt.title(architectures[i])

        plt.plot(np.arange(0, attempts), np.repeat(child_reaction['anger'], attempts), 'r--')
        plt.plot(np.arange(0, attempts), np.repeat(child_reaction['sadness'], attempts), 'b--')
        plt.plot(np.arange(0, attempts), np.repeat(child_reaction['happy'], attempts), 'g--')

        plt.plot(np.arange(0, attempts), anger[i::3], 'r', )
        plt.plot(np.arange(0, attempts), sadness[i::3], 'b')
        plt.plot(np.arange(0, attempts), happy[i::3], 'g')
        plt.legend()

    # plt.title(architectures[0])
    #
    # plt.plot(np.arange(0, attempts), np.repeat(child_reaction['anger'], attempts), 'r--')
    # plt.plot(np.arange(0, attempts), np.repeat(child_reaction['sadness'], attempts), 'b--')
    # plt.plot(np.arange(0, attempts), np.repeat(child_reaction['happy'], attempts), 'g--')
    #
    # plt.plot(np.arange(0, attempts), anger, 'r', label='anger')
    # plt.plot(np.arange(0, attempts), sadness, 'b', label='sadness')
    # plt.plot(np.arange(0, attempts), happy, 'g', label='happy')
    # plt.legend()

    plt.savefig('.'.join([file_name.split('.')[0], 'png']))
    plt.close()

# plt.show()


if __name__ == '__main__':

    print 'Experiment 1: Interaction between happy/sad children and Secure Parent'
    print 'Experiment 2: Interaction between happy/sad children and Ambivalent Parent'
    print 'Experiment 3: Interaction between happy/sad children and Avoidant Parent'

    secure_mapping = ({'happy': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                       'sadness': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                       'anger': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                       })
    ambivalent_mapping = ({'happy': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                           'anger': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                           'sadness': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                           })
    avoidant_mapping = ({'happy': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                         'anger': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                         'sadness': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                         })

    # plot_result('data/remote/Kanade/Experiment1/Experiment1.txt', mapping)
    # plot_result('data/remote/Kanade/Experiment1_50/Experiment1_50.txt', mapping)

    def experiment_child(proj_name, mapping, shape):
        attempt = 10
        for i in xrange(0, attempt):
            print 'attempt %d' % i
            # experimentChild('Experiment4', mapping, 25, 'rbm')
            experimentChild(proj_name, mapping, shape, 'dbn')
            experimentChild(proj_name, mapping, shape, 'adbn')
            # experimentChild('Experiment4', mapping, 25, 'adbn')

    # experiment_child('Experiment7', secure_mapping, 25)
    # experiment_child('Experiment7_50', secure_mapping, 50)
    # experiment_child('Experiment8', ambivalent_mapping, 25)
    # experiment_child('Experiment8_50', ambivalent_mapping, 50)
    experiment_child('Experiment9', avoidant_mapping, 25)
    # experiment_child('Experiment9_50', avoidant_mapping, 50)

    # plot_result('data/Experiment4_50/Experiment4_50.txt', mapping)

    attempt = 0
    #
    for i in xrange(0, attempt):
        # experimentChild('Experiment5', mapping, 25, 'rbm')
        experimentChild('Experiment8', ambivalent_mapping, 25, 'dbn')
        experimentChild('Experiment8', ambivalent_mapping, 25, 'adbn')
        # experimentChild('Experiment5', mapping, 25, 'adbn')

    #
    # plot_result('data/Experiment2/Experiment2.txt', mapping)
    # plot_result('data/Experiment2_50/Experiment2_50.txt',mapping)
    # plot_result('data/remote/Kanade/Experiment2/Experiment2.txt', mapping)
    # plot_result('data/remote/Kanade/Experiment2_50/Experiment2_50.txt', mapping)



    for i in xrange(0, attempt):
        # experimentChild('Experiment6', mapping, 25, 'rbm')
        experimentChild('Experiment9', avoidant_mapping, 25, 'dbn')
        experimentChild('Experiment9', avoidant_mapping, 25, 'adbn')
        # experimentChild('Experiment6', mapping, 25, 'adbn')

    for i in xrange(0, attempt):
        # experimentChild('Experiment4_50', mapping, 50, 'rbm')
        experimentChild('Experiment7_50', secure_mapping, 50, 'dbn')
        experimentChild('Experiment7_50', secure_mapping, 50, 'adbn')
        # experimentChild('Experiment4_50', mapping, 50, 'adbn')

    for i in xrange(0, attempt):
        # experimentChild('Experiment5_50', mapping, 50, 'rbm')
        experimentChild('Experiment8_50', ambivalent_mapping, 50, 'adbn')
        # experimentChild('Experiment5_50', mapping, 50, 'adbn')

    for i in xrange(0, attempt):
        # experimentChild('Experiment6_50', mapping, 50, 'rbm')
        experimentChild('Experiment9_50', avoidant_mapping, 50, 'dbn')
        # experimentChild('Experiment6_50', mapping, 50, 'adbn')

        # plot_result('data/Experiment3/Experiment3.txt',mapping)
        # plot_result('data/Experiment3_50/Experiment3_50.txt',mapping)
        # plot_result('data/remote/Kanade/Experiment6/Experiment6.txt', mapping)
        # plot_result('data/remote/Kanade/Experiment3/Experiment3.txt', mapping)

