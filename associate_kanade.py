__author__ = 'joschlemper'

import rbm as RBM
import associative_dbn
import kanade_loader as loader
import datastorage as store
import utils
import theano


def train_kanade():
    print "Testing RBM"

    data_manager = store.StorageManager('Kanade/SimpleRBMTest')

    # Load mnist hand digits
    datasets = loader.load_kanade(n=500, resolution='50_50', emotions=['happy','sadness'], pre={'scale2unit': True})
    train_x, train_y = datasets

    sparsity_constraint = False
    # Initialise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.0005,
                        momentum_type=RBM.NESTEROV,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=sparsity_constraint,
                        sparsity_target=0.001,
                        sparsity_cost=1,
                        sparsity_decay=0.9,
                        epochs=100)

    n_visible = train_x.get_value().shape[1]
    n_hidden = 100

    rbm = RBM.RBM(n_visible,
                  n_visible,
                  n_hidden,
                  associative=False,
                  dropout=True,
                  cd_type=RBM.CLASSICAL,
                  cd_steps=1,
                  train_parameters=tr,
                  progress_logger=RBM.ProgressLogger(img_shape=(50, 50)))

    print "... initialised RBM"

    if sparsity_constraint:
        rbm.get_initial_mean_activity(train_x)


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

    resolution='25_25'
    img_shape = (25, 25)

    # Load mnist hand digits, class label is already set to binary
    dataset = loader.load_kanade(n=500, resolution=resolution, emotions=['anger','happy','sadness'], pre={'scale2unit': True})
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
    reconstructed_y = rbm.reconstruct_association(train_x, None, 30, 0.0, plot_n=100, plot_every=1, img_name='kanade_recon.png')
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
                ass_dbn.train(train_x, train_x01, cache=cache)#, optimise=True)

                for n_recall in [1, 3, 10]:
                    for n_think in [0, 1, 3, 7]: #1, 3, 5, 7, 10]:
                        # Reconstruct
                        sampled = ass_dbn.recall(train_x, n_recall, n_think)

                        # Sample from top layer to generate data
                        sample_n = 100
                        utils.save_digits(sampled, image_name='{}_reconstruced_{}_{}_{}.png'.format(count, n_ass, n_recall, n_think), shape=(sample_n / 10, 10),img_shape=(25, 25))
                        count += 1


if __name__ == '__main__':
    # train_kanade()
    # associate_data2data()
    associate_data2dataDBN()