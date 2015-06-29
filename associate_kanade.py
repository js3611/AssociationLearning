__author__ = 'joschlemper'

from models.rbm import RBM
from models import rbm_logger, rbm_config, rbm_units, DBN, associative_dbn
import kanade_loader as loader
import datastorage as store
import utils
import numpy as np
import theano
import theano.tensor as T
from simple_classifiers import SimpleClassifier


def train_kanade():
    print "Testing RBM"

    data_manager = store.StorageManager('Kanade/SimpleRBMTest')

    # Load mnist hand digits
    datasets = loader.load_kanade(n=500, set_name='25_25', emotions=['happy', 'sadness'], pre={'scale2unit': True})
    train_x, train_y = datasets[0]

    sparsity_constraint = True
    # Initialise the RBM and training parameters
    tr = rbm_config.TrainParam(learning_rate=0.0001,
                               momentum_type=rbm_config.NESTEROV,
                               momentum=0.9,
                               weight_decay=0.001,
                               sparsity_constraint=sparsity_constraint,
                               sparsity_target=0.01,
                               sparsity_cost=1,
                               sparsity_decay=0.9,
                               epochs=100)

    n_visible = train_x.get_value().shape[1]
    n_hidden = 500

    config = rbm_config.RBMConfig(n_visible,
                                  n_visible,
                                  n_hidden,
                                  v_unit=rbm_units.GaussianVisibleUnit,
                                  associative=False,
                                  dropout=True,
                                  cd_type=rbm_config.CLASSICAL,
                                  cd_steps=1,
                                  train_parameters=tr,
                                  progress_logger=RBM.ProgressLogger(img_shape=(25, 25)))

    rbm = RBM(config)

    print "... initialised RBM"

    if sparsity_constraint:
        rbm.set_initial_hidden_bias()
        rbm.set_hidden_mean_activity(train_x)


    # adjust learning rate
    # rbm.pretrain_lr(train_x)
    # rbm.pretrain_mean_activity_h(train_x)

    # Train RBM
    rbm.train(train_x)

    # Test RBM
    rbm.reconstruct(train_x, k=5, plot_n=10, plot_every=1)

    # Store Parameters
    data_manager.persist(rbm)


def associate_data2data(cache=False):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('Kanade/AssociationRBMTest', log=True)

    resolution = '25_25'
    img_shape = (25, 25)

    # Load mnist hand digits, class label is already set to binary
    dataset = loader.load_kanade(n=500, set_name=resolution, emotions=['anger', 'happy', 'sadness'],
                                 pre={'scale2unit': True})
    train_x, train_y = dataset
    # for now id map
    train_x01 = loader.sample_image(train_y)

    dataset01 = loader.load_kanade(n=10)


    # Initialise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.001,
                        momentum_type=RBM.CLASSICAL,
                        momentum=0.0,
                        weight_decay=0.000,
                        sparsity_constraint=False,
                        sparsity_target=0.1 ** 9,
                        sparsity_cost=0.9,
                        sparsity_decay=0.99,
                        epochs=100)
    # tr = RBM.TrainParam(learning_rate=0.005,
    # momentum_type=RBM.CLASSICAL,
    # momentum=0.5,
    # weight_decay=0.001,
    # sparsity_constraint=False,
    # sparsity_target=0.01,
    # sparsity_cost=0.5,
    # sparsity_decay=0.9,
    # epochs=20)

    # Even odd test
    k = 1
    n_visible = train_x.get_value().shape[1]
    n_visible2 = n_visible
    n_hidden = 100

    rbm = RBM.RBM(n_visible,
                  n_visible2,
                  n_hidden,
                  associative=True,
                  cd_type=RBM.CLASSICAL,
                  cd_steps=k,
                  train_parameters=tr,
                  progress_logger=RBM.AssociationProgressLogger(img_shape=img_shape))

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded and cache:
        rbm = loaded
        print "... loaded precomputed rbm"
    else:
        # Train RBM - learn joint distribution
        # rbm.pretrain_lr(train_x, train_x01)
        rbm.train(train_x, train_x01)
        rbm.save()

    print "... reconstruction of associated images"
    reconstructed_y = rbm.reconstruct_association(train_x, None, 30, 0.0, plot_n=100, plot_every=1,
                                                  img_name='kanade_recon.png')
    print "... reconstructed"

    # TODO use sklearn to obtain accuracy/precision etc

    # Create Dataset to feed into logistic regression
    # Test set: reconstructed y's become the input. Get the corresponding x's and y's
    # dataset01[2] = (theano.shared(reconstructed_y), test_y)
    #
    # # Classify the reconstructions
    # score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset01, 100)
    #
    # print 'Score: {}'.format(str(score))
    # print str(rbm)


def KanadeAssociativeOptRBM(cache=False, train_further=False):
    print "Testing Associative RBM which tries to learn the following mapping: {anger, saddness, disgust} -> {sadness}, {contempt, happy, surprise} -> {happy}"
    # project set-up
    data_manager = store.StorageManager('Kanade/OptMFSparse0.01RBMTest', log=True)
    # data_manager = store.StorageManager('Kanade/OptAssociativeRBMTest', log=True)
    shape = 25
    dataset_name = 'sharp_equi{}_{}'.format(shape, shape)

    # Load kanade database
    mapping = None  # id map
    # mapping = {'anger': 'sadness', 'contempt': 'happy', 'disgust': 'sadness', 'fear': 'sadness', 'happy': 'happy',
    #            'sadness': 'sadness', 'surprise': 'happy'}
    train, valid, test = loader.load_kanade(pre={'scale': True}, set_name=dataset_name)
    train_x, train_y = train
    test_x, test_y = test

    # Sample associated image
    train_x_mapped, train_y_mapped = loader.sample_image(train_y, mapping=mapping, pre={'scale': True},
                                                         set_name=dataset_name)
    test_x_mapped, test_y_mapped = loader.sample_image(test_y, mapping=mapping, pre={'scale': True},
                                                       set_name=dataset_name)

    # Concatenate images
    concat1 = T.concatenate([train_x, train_x_mapped], axis=1)
    # concat2 = T.concatenate([train_x_mapped, train_x], axis=1)
    # concat = T.concatenate([concat1, concat2], axis=0)
    # train_tX = theano.function([], concat)()
    train_tX = theano.function([], concat1)()
    train_X = theano.shared(train_tX)

    # Train classifier to be used for classifying reconstruction associated image layer
    # mapped_data = loader.load_kanade(#emotions=['sadness', 'happy'],
    #                                  pre={'scale': True},
    #                                  set_name=dataset_name)  # Target Image
    # clf_orig = SimpleClassifier('logistic', mapped_data[0][0], mapped_data[0][1])
    clf_orig = SimpleClassifier('logistic', train_x, train_y)

    # Initialise RBM
    tr = rbm_config.TrainParam(learning_rate=0.0001,
                               momentum_type=rbm_config.NESTEROV,
                               momentum=0.9,
                               weight_decay=0.0001,
                               sparsity_constraint=True,
                               sparsity_target=0.01,
                               sparsity_cost=100,
                               sparsity_decay=0.9,
                               batch_size=10,
                               epochs=10)

    n_visible = shape * shape * 2
    n_hidden = 500

    config = rbm_config.RBMConfig()
    config.v_n = n_visible
    config.h_n = n_hidden
    config.v_unit = rbm_units.GaussianVisibleUnit
    # config.h_unit = rbm_units.ReLUnit
    config.progress_logger = rbm_logger.ProgressLogger(img_shape=(shape * 2, shape))
    config.train_params = tr
    rbm = RBM(config)
    print "... initialised RBM"

    # Load RBM (test)
    loaded = data_manager.retrieve(str(rbm))
    if loaded:
        rbm = loaded
    else:
        rbm.set_initial_hidden_bias()
        rbm.set_hidden_mean_activity(train_X)

    # Train RBM - learn joint distribution
    # rbm.pretrain_lr(train_x, train_x01)
    for i in xrange(0, 10):
        if not cache or train_further:
            rbm.train(train_X)

        data_manager.persist(rbm)

        print "... reconstruction of associated images"
        # Get reconstruction with train data to get 'mapped' images to train classifiers on
        reconstruction = rbm.reconstruct(train_X, 1,
                                         plot_n=100,
                                         plot_every=1,
                                         img_name='recon_train')
        reconstruct_assoc_part = reconstruction[:, (shape ** 2):]

        # Get associated images of test data
        nsamples = np.random.normal(0, 1, test_x.get_value(True).shape).astype(np.float32)
        # nsamples = np.random.normal(0, 0.0001, test_x.get_value(True).shape)
        # nsamples = np.random.normal(0, 1, test_x.get_value(True).shape)
        initial_y = theano.shared(nsamples, name='initial_y')
        utils.save_images(nsamples[0:100], 'initialisation.png', (10, 10), (25, 25))

        test_x_associated = rbm.reconstruct_association_opt(test_x, initial_y,
                                                            5,
                                                            0.,
                                                            plot_n=100,
                                                            plot_every=1,
                                                            img_name='recon_test_gibbs')

        mf_recon = rbm.mean_field_inference_opt(test_x, y=initial_y,
                                                sample=False,
                                                k=10,
                                                img_name='recon_test_mf_raw')

        # Concatenate images
        test_MFX = theano.function([], T.concatenate([test_x, mf_recon], axis=1))()
        test_MF = theano.shared(test_MFX)
        reconstruction = rbm.reconstruct(test_MF, 1,
                                         plot_n=100,
                                         plot_every=1,
                                         img_name='recon_test_mf_recon')
        mf_recon = reconstruction[:, (shape ** 2):]

        print "... reconstructed"

        # Classify the reconstructions

        # 1. Train classifier on original images
        score_orig = clf_orig.get_score(test_x_associated, test_y_mapped.eval())
        score_orig_mf = clf_orig.get_score(test_x_associated, test_y_mapped.eval())

        # 2. Train classifier on reconstructed images
        clf_recon = SimpleClassifier('logistic', reconstruct_assoc_part, train_y_mapped.eval())
        score_retrain = clf_recon.get_score(test_x_associated, test_y_mapped.eval())
        score_retrain_mf = clf_recon.get_score(mf_recon, test_y_mapped.eval())

        out_msg = '{} (orig, retrain):{},{}'.format(rbm, score_orig, score_retrain)
        out_msg2 = '{} (orig, retrain):{},{}'.format(rbm, score_orig_mf, score_retrain_mf)
        print out_msg
        print out_msg2


def KanadeAssociativeDBN(cache=False):
    print "Testing Associative RBM which tries to learn the following mapping: " \
          "{anger, saddness, disgust} -> {sadness}, {contempt, happy, surprise} -> {happy}"
    # project set-up
    data_manager = store.StorageManager('Kanade/AssociativeDBNTest', log=True)
    shape = 25
    dataset_name = 'sharp_equi{}_{}'.format(shape, shape)
    preprocessing = {'scale': True}

    # Load kanade database
    mapping = None
    # mapping = {'anger': 'sadness',
    #            'contempt': 'happy',
    #            'disgust': 'sadness',
    #            'fear': 'sadness',
    #            'happy': 'happy',
    #            'sadness': 'sadness',
    #            'surprise': 'happy'}

    dataset = loader.load_kanade(  # n=3000,
                                   pre=preprocessing,
                                   set_name=dataset_name)

    mapped_dataset = loader.load_kanade(  # n=3000,
                                          # emotions=['sadness', 'happy'],
                                          pre=preprocessing,
                                          set_name=dataset_name)  # Target Image
    train, valid, test = dataset
    train_x, train_y = train
    test_x, test_y = test

    # Sample associated image
    train_x_ass, train_y_ass = loader.sample_image(train_y,
                                                   mapping=mapping,
                                                   pre=preprocessing,
                                                   set_name=dataset_name)
    test_x_ass, test_y_ass = loader.sample_image(test_y,
                                                 mapping=mapping,
                                                 pre=preprocessing,
                                                 set_name=dataset_name)

    # initialise AssociativeDBN
    config = associative_dbn.DefaultADBNConfig()

    # Gaussian Input Layer
    bottom_tr = rbm_config.TrainParam(learning_rate=0.0001,
                                      momentum_type=rbm_config.NESTEROV,
                                      momentum=0.9,
                                      weight_decay=0.0001,
                                      epochs=20,
                                      batch_size=10)
    h_n = 150
    bottom_logger = rbm_logger.ProgressLogger(img_shape=(shape, shape))
    bottom_rbm = rbm_config.RBMConfig(v_unit=rbm_units.GaussianVisibleUnit,
                                      v_n=shape ** 2,
                                      h_n=h_n,
                                      progress_logger=bottom_logger,
                                      train_params=bottom_tr)

    config.left_dbn.rbm_configs[0] = bottom_rbm
    config.right_dbn.rbm_configs[0] = bottom_rbm
    config.left_dbn.topology = [shape ** 2, h_n]
    config.right_dbn.topology = [shape ** 2, h_n]
    config.top_rbm.train_params.epochs = 20
    config.top_rbm.train_params.batch_size = 10
    config.n_association = 1000
    config.reuse_dbn = True
    adbn = associative_dbn.AssociativeDBN(config=config, data_manager=data_manager)

    # Plot sample
    loader.save_faces(train_x.get_value(borrow=True)[1:50], tile=(10, 10), img_name='n_orig.png', )
    loader.save_faces(train_x_ass.get_value(borrow=True)[1:50], tile=(10, 10), img_name='n_ass.png')

    # Train classifier to be used for classifying reconstruction associated image layer
    clf_orig = SimpleClassifier('knn', mapped_dataset[0][0], mapped_dataset[0][1])

    # Test DBN Performance
    for i in xrange(0, 5):
        # Train DBN - learn joint distribution
        cache_left = [True]
        cache_right = [True]
        cache_top = False
        cache = [cache_left, cache_right, cache_top]
        adbn.train(train_x, train_x_ass, cache=cache)
        print "... trained associative DBN"

        # Reconstruct images
        test_x_recon = adbn.recall(test_x, associate_steps=500, recall_steps=0)
        print "... reconstructed images"


        # Classify the reconstructions

        # 1. Train classifier on original images
        score_orig = clf_orig.get_score(test_x_recon, test_y_ass.eval())

        # 2. Train classifier on reconstructed images - reconstruction obtained by right dbn
        right_dbn = adbn.dbn_right
        mapped_train_recon = right_dbn.reconstruct(mapped_dataset[0][0],
                                                   k=1,
                                                   plot_n=100,
                                                   plot_every=1,
                                                   img_name='right_dbn_reconstruction')
        clf_recon = SimpleClassifier('knn', mapped_train_recon, mapped_dataset[0][1].eval())
        score_retrain = clf_recon.get_score(test_x_recon, test_y_ass.eval())

        out_msg = '{} (orig, retrain):{},{}'.format(adbn, score_orig, score_retrain)
        print out_msg


def KanadeStackedOptRBM(cache=False):
    print "Testing Stacked-RBM which tries to learn id map association"

    # project set-up
    data_manager = store.StorageManager('Kanade/MeanField3', log=True)
    shape = 25
    dataset_name = 'sharp_equi{}_{}'.format(shape, shape)
    preprocessing = {'scale': True}

    # Load kanade database
    mapping = None
    # mapping = {'anger': 'sadness',
    #            'contempt': 'happy',
    #            'disgust': 'sadness',
    #            'fear': 'sadness',
    #            'happy': 'happy',
    #            'sadness': 'sadness',
    #            'surprise': 'happy'}

    dataset = loader.load_kanade(  # n=3000,
                                   pre=preprocessing,
                                   set_name=dataset_name)

    mapped_dataset = loader.load_kanade(  # n=3000,
                                          # emotions=['sadness', 'happy'],
                                          pre=preprocessing,
                                          set_name=dataset_name)  # Target Image
    train, valid, test = dataset
    train_x, train_y = train
    test_x, test_y = test

    # Sample associated image
    train_x_ass, train_y_ass = loader.sample_image(train_y,
                                                   mapping=mapping,
                                                   pre=preprocessing,
                                                   set_name=dataset_name)
    test_x_ass, test_y_ass = loader.sample_image(test_y,
                                                 mapping=mapping,
                                                 pre=preprocessing,
                                                 set_name=dataset_name)

    # Initialise RBM parameters
    base_tr = rbm_config.TrainParam(learning_rate=0.0001,
                                    momentum_type=rbm_config.NESTEROV,
                                    momentum=0.9,
                                    weight_decay=0.0001,
                                    sparsity_constraint=False,
                                    sparsity_target=0.00001,
                                    sparsity_decay=0.9,
                                    sparsity_cost=10000,
                                    epochs=100,
                                    batch_size=10)

    rest_tr = rbm_config.TrainParam(learning_rate=0.0001,
                                    momentum_type=rbm_config.CLASSICAL,
                                    momentum=0.5,
                                    weight_decay=0.01,
                                    epochs=100,
                                    batch_size=10)

    # Layer 1
    # Layer 2
    # Layer 3
    topology = [2 * (shape ** 2), 3000]
    # batch_size = 10
    first_progress_logger = rbm_logger.ProgressLogger(img_shape=(shape * 2, shape))
    rest_progress_logger = rbm_logger.ProgressLogger()

    first_rbm_config = rbm_config.RBMConfig(train_params=base_tr,
                                            progress_logger=first_progress_logger)
    first_rbm_config.v_unit = rbm_units.GaussianVisibleUnit
    rest_rbm_config = rbm_config.RBMConfig(train_params=rest_tr,
                                           progress_logger=rest_progress_logger)
    rbm_configs = [first_rbm_config, rest_rbm_config, rest_rbm_config]

    config = DBN.DBNConfig(topology=topology,
                           training_parameters=base_tr,
                           rbm_configs=rbm_configs,
                           data_manager=data_manager)

    # construct the Deep Belief Network
    dbn = DBN.DBN(config)

    # Train DBN on concatenated images
    train_tX = theano.function([], T.concatenate([train_x, train_x_ass], axis=1))()
    train_X = theano.shared(train_tX)
    test_tX = theano.function([], T.concatenate([test_x, test_x_ass], axis=1))()
    test_X = theano.shared(test_tX)
    test_tX2 = theano.function([], T.concatenate([test_x, T.zeros_like(test_x)], axis=1))()
    test_X2 = theano.shared(test_tX2)

    origs = []
    recons = []
    recons2 = []

    # Train DBN
    dbn.pretrain(train_X, cache=[True, True, False], train_further=[True, True, True], optimise=False)

    recon = dbn.reconstruct(train_X, k=1, plot_n=20,
                            img_name='stackedRBM_train_recon_{}_{}'.format(topology, 0))
    train_x_ass_recon = recon[:, shape ** 2:]

    recon = dbn.reconstruct(test_X, k=1, plot_n=20,
                            img_name='stackedRBM_test_recon_{}_{}'.format(topology, 0))
    test_x_ass_recon = recon[:, shape ** 2:]

    recon = dbn.reconstruct(test_X2, k=2, plot_n=20,
                            img_name='stackedRBM_test_zero_recon_{}_{}'.format(topology, 0))
    test_x_ass_recon2 = recon[:, shape ** 2:]

    clf_recon = SimpleClassifier('logistic', train_x, train_y)
    score_orig = clf_recon.get_score(test_x_ass_recon, test_y_ass.eval())

    clf_recon.retrain(train_x_ass_recon, train_y_ass.eval())
    score_recon = clf_recon.get_score(test_x_ass_recon, test_y_ass.eval())
    score_recon2 = clf_recon.get_score(test_x_ass_recon2, test_y_ass.eval())

    print 'classification rate: {}, {}, {}'.format(score_orig, score_recon, score_recon2)
    origs.append(score_orig)
    recons.append(score_recon)
    recons2.append(score_recon2)


def associate_data2dataDBN(cache=False):
    print "Testing Associative DBN which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('Kanade/associative_dbn_test', log=True)


    # Load mnist hand digits, class label is already set to binary
    dataset = loader.load_kanade(n=500, emotions=['anger', 'sadness', 'happy'], pre={'scale2unit': True})
    train_x, train_y = dataset
    train_x01 = loader.sample_image(train_y)

    dataset01 = loader.load_kanade(n=500)

    # Initialise RBM parameters
    # fixed base train param
    base_tr = RBM.TrainParam(learning_rate=0.001,
                             momentum_type=RBM.CLASSICAL,
                             momentum=0.5,
                             weight_decay=0.0005,
                             sparsity_constraint=False,
                             epochs=20)

    # top layer parameters
    tr = RBM.TrainParam(learning_rate=0.001,
                        # find_learning_rate=True,
                        momentum_type=RBM.NESTEROV,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=False,
                        epochs=20)

    tr_top = RBM.TrainParam(learning_rate=0.001,
                            # find_learning_rate=True,
                            momentum_type=RBM.CLASSICAL,
                            momentum=0.5,
                            weight_decay=0.001,
                            sparsity_constraint=False,
                            epochs=20)


    # Layer 1
    # Layer 2
    # Layer 3
    # topology = [784, 500, 500, 100]

    config = associative_dbn.DefaultADBNConfig()
    config.topology_left = [625, 500, 500, 100]
    config.topology_right = [625, 500, 500, 100]
    config.reuse_dbn = False
    config.top_rbm_params = tr_top
    config.base_rbm_params = [base_tr, tr, tr]

    count = 0
    for cd_type in [RBM.CLASSICAL, RBM.PERSISTENT]:
        for n_ass in [100, 250, 500, 750, 1000]:
            config.n_association = n_ass
            config.top_cd_type = cd_type

            # Construct DBN
            ass_dbn = associative_dbn.AssociativeDBN(config=config, data_manager=data_manager)

            # Train
            for trainN in xrange(0, 5):
                ass_dbn.train(train_x, train_x01, cache=cache)  # , optimise=True)

                for n_recall in [1, 3, 10]:
                    for n_think in [0, 1, 3, 5, 10]:  # 1, 3, 5, 7, 10]:
                        # Reconstruct
                        sampled = ass_dbn.recall(train_x, n_recall, n_think)

                        # Sample from top layer to generate data
                        sample_n = 100
                        utils.save_images(sampled,
                                          image_name='{}_reconstruced_{}_{}_{}.png'.format(count, n_ass, n_recall,
                                                                                           n_think),
                                          shape=(sample_n / 10, 10), img_shape=(25, 25))
                        count += 1


if __name__ == '__main__':
    # train_kanade()
    # associate_data2data()
    KanadeAssociativeOptRBM(True, train_further=True)
    # KanadeAssociativeDBN()
    # KanadeStackedOptRBM()
