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
                                          pre=preprocesssing)
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
        brain_c.train(tr_x)
        # brain_c.train(tr_x_mixed)

        recon = brain_c.reconstruct_association_opt(h_te_x, initial_y,
                                                    k=10,
                                                    plot_n=100,
                                                    img_name='rbm_child_recon_{}'.format(shape))

        recon_pair = brain_c.reconstruct(tr_x, k=1, plot_n=100, img_name='rbm_pair_recon_{}'.format(shape))
        recon_p_tr_x = recon_pair[:, (shape ** 2):]


    elif model == 'dbn':
        brain_c = get_brain_model_DBN(shape, data_manager)
        brain_c.pretrain(tr_x, cache=[True, True, True])

        recon_pair = brain_c.reconstruct(tr_x, k=1, plot_n=100, img_name='dbn_pair_recon_{}'.format(shape))
        recon_p_tr_x = recon_pair[:, (shape ** 2):]

        recon_pair = brain_c.reconstruct(te_x, k=10, plot_n=100, img_name='dbn_child_recon_{}'.format(shape))
        recon = recon_pair[:, (shape ** 2):]

    elif model == 'adbn':
        brain_c = get_brain_model_AssociativeDBN(shape, data_manager)
        brain_c.train(h_tr_x, p_tr_x, cache=[[True, True, True], [True, True, True], True])

        # Reconstruction
        recon_p_tr_x = brain_c.dbn_right.reconstruct(p_tr_x, k=1, plot_n=100, img_name='adbn_right_recon_{}'.format(shape))
        recon = brain_c.recall(h_te_x, associate_steps=1, recall_steps=1, img_name='adbn_child_recon_{}'.format(shape))

    # Train classifier on reconstruction
    clf = SimpleClassifier('logistic', recon_p_tr_x, p_tr_y.eval())

    # Output number of classes
    res = clf.classify(recon).T
    plt.plot(res)
    plt.ylabel('Classify Count')
    plt.show()
    r = itemfreq(res).T
    labels = map(lambda x: kanade_loader.emotion_rev_dict[int(x)], r[0])
    proportion = r[1] / sum(r[1])
    txt = 'Learnt Child Configuration:'
    print txt
    f.write(txt)
    f.write('\n')
    for i, l in enumerate(labels):
        fill_space = (max(map(lambda x: len(x), labels))-len(l))
        txt = '{}{}:\t %.3f'.format(l, ' ' * fill_space) % proportion[i]
        print txt
        f.write(txt)
        f.write('\n')
    f.write('\n')
    f.close()

def get_brain_model_RBM(shape):
    # Initialise RBM
    tr = TrainParam(learning_rate=0.0001,
                    momentum_type=NESTEROV,
                    momentum=0.9,
                    weight_decay=0.0001,
                    sparsity_constraint=False,
                    batch_size=10,
                    epochs=20)
    n_visible = shape * shape * 2
    n_hidden = 250
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
                           epochs=20,
                           batch_size=10)
    h_n = 250
    bottom_logger = ProgressLogger(img_shape=(shape, shape))
    bottom_rbm = RBMConfig(v_unit=rbm_units.GaussianVisibleUnit,
                           v_n=shape ** 2,
                           h_n=h_n,
                           progress_logger=bottom_logger,
                           train_params=bottom_tr)

    config.left_dbn.rbm_configs[0] = bottom_rbm
    config.right_dbn.rbm_configs[0] = bottom_rbm
    config.left_dbn.topology = [shape ** 2, h_n, 250,]
    config.right_dbn.topology = [shape ** 2, h_n, 250,]
    config.top_rbm.train_params.epochs = 20
    config.top_rbm.train_params.batch_size = 10
    config.n_association = 500
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
                         sparsity_target=0.00001,
                         sparsity_decay=0.9,
                         sparsity_cost=10000,
                         epochs=20,
                         batch_size=10)

    rest_tr = TrainParam(learning_rate=0.0001,
                         momentum_type=CLASSICAL,
                         momentum=0.5,
                         weight_decay=0.01,
                         epochs=20,
                         batch_size=10)

    # Layer 1
    # Layer 2
    # Layer 3
    topology = [2 * (shape ** 2), 250, 250, 500]
    # batch_size = 10
    first_progress_logger = ProgressLogger(img_shape=(shape * 2, shape))
    rest_progress_logger = ProgressLogger()

    first_rbm_config = RBMConfig(train_params=base_tr,
                                 progress_logger=first_progress_logger)
    first_rbm_config.v_unit = rbm_units.GaussianVisibleUnit
    rest_rbm_config = RBMConfig(train_params=rest_tr,
                                progress_logger=rest_progress_logger)
    rbm_configs = [first_rbm_config, rest_rbm_config, rest_rbm_config]

    config = DBN.DBNConfig(topology=topology,
                           training_parameters=base_tr,
                           rbm_configs=rbm_configs,
                           data_manager=data_manager)

    # construct the Deep Belief Network
    dbn = DBN.DBN(config)
    print '... initialised DBN'
    return dbn


if __name__ == '__main__':
    print 'Experiment 1: Interaction between happy/sad children and Secure Parent'
    mapping = ({'happy': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                'sadness': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                'anger': {'happy': 0.8, 'anger': 0.1, 'sadness': 0.1},
                })

    experimentChild('Experiment1', mapping, 25, 'rbm')
    experimentChild('Experiment1', mapping, 25, 'dbn')
    experimentChild('Experiment1', mapping, 25, 'adbn')
    experimentChild('Experiment1', mapping, 50, 'rbm')
    experimentChild('Experiment1', mapping, 50, 'dbn')
    experimentChild('Experiment1', mapping, 50, 'adbn')

    print 'Experiment 2: Interaction between happy/sad children and Ambivalent Parent'
    mapping = ({'happy': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                'anger': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                'sadness': {'happy': 0.5, 'anger': 0.2, 'sadness': 0.3},
                })

    experimentChild('Experiment2', mapping, 25, 'rbm')
    experimentChild('Experiment2', mapping, 25, 'dbn')
    experimentChild('Experiment2', mapping, 25, 'adbn')
    experimentChild('Experiment2', mapping, 50, 'rbm')
    experimentChild('Experiment2', mapping, 50, 'dbn')
    experimentChild('Experiment2', mapping, 50, 'adbn')

    print 'Experiment 3: Interaction between happy/sad children and Avoidant Parent'
    mapping = ({'happy': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                'anger': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                'sadness': {'happy': 0.3, 'anger': 0.5, 'sadness': 0.2},
                })

    experimentChild('Experiment3', mapping, 25, 'rbm')
    experimentChild('Experiment3', mapping, 25, 'dbn')
    experimentChild('Experiment3', mapping, 25, 'adbn')
    experimentChild('Experiment3', mapping, 50, 'rbm')
    experimentChild('Experiment3', mapping, 50, 'dbn')
    experimentChild('Experiment3', mapping, 50, 'adbn')



