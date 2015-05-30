import theano
import theano.tensor as T
import numpy as np
import rbm as RBM
import DBN
import associative_dbn
import utils
import m_loader as m_loader
import datastorage as store

import logging

logging.basicConfig(filename='trace.log', level=logging.INFO)

def associate_data2label(cache=False):
    print "Testing ClassRBM with generative target (i.e. AssociativeRBM with picture-label association)"

    # Load mnist hand digits, class label is already set to binary
    train, valid, test = m_loader.load_digits(n=[500, 100, 100], pre={'label_vector': True})
    train_x, train_y = train
    test_x, test_y = test
    train_y = T.cast(train_y, dtype=theano.config.floatX)
    test_y = T.cast(test_y, dtype=theano.config.floatX)

    # Initialise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.1,
                        momentum_type=RBM.CLASSICAL,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=True,
                        sparsity_target=0.01,
                        sparsity_cost=0.5,
                        sparsity_decay=0.9,
                        epochs=20)
    n_visible = train_x.get_value().shape[1]
    n_visible2 = 10
    n_hidden = 500

    rbm = RBM.RBM(n_visible,
                  n_visible2,
                  n_hidden,
                  associative=True,
                  cd_type=RBM.CLASSICAL,
                  cd_steps=1,
                  train_parameters=tr,
                  progress_logger=RBM.ProgressLogger())

    store.move_to('label_test/'+str(rbm))

    loaded = store.retrieve_object(str(rbm))
    if loaded and cache:
        rbm = loaded
        print "... loaded precomputed rbm"
    else:
        rbm.train(train_x, train_y)
        rbm.save()

    rbm.associative = False
    rbm.reconstruct(test_x, 10)
    rbm.associative = True

    # Classification test - Reconstruct y through x
    output_probability = rbm.reconstruct_association(test_x, None)
    sol = np.array([np.argmax(lab) for lab in test_y.eval()])
    guess = np.array([np.argmax(lab) for lab in output_probability])
    diff = np.count_nonzero(sol == guess)

    print "Classification Rate: {}".format((diff / float(test_y.eval().shape[0])))


def associate_data2data(cache=False):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('AssociationRBMTest', log=True)


    # Load mnist hand digits, class label is already set to binary
    train, valid, test = m_loader.load_digits(n=[500, 100, 500], digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = m_loader.sample_image(train_y)

    dataset01 = m_loader.load_digits(n=[500, 100, 100], digits=[0, 1])


    # Initialise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.1,
                        momentum_type=RBM.CLASSICAL,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=True,
                        sparsity_target=0.1 ** 9,
                        sparsity_cost=0.9,
                        sparsity_decay=0.99,
                        epochs=50)
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
    n_hidden = 200

    rbm = RBM.RBM(n_visible,
                  n_visible2,
                  n_hidden,
                  associative=True,
                  cd_type=RBM.CLASSICAL,
                  cd_steps=k,
                  train_parameters=tr,
                  progress_logger=RBM.AssociationProgressLogger())

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded and cache:
        rbm = loaded
        print "... loaded precomputed rbm"
    else:
        # Train RBM - learn joint distribution
        rbm.pretrain_lr(train_x, train_x01)
        rbm.train(train_x, train_x01)
        rbm.save()

    print "... reconstruction of associated images"
    reconstructed_y = rbm.reconstruct_association(test_x, None, 30, 0.01, plot_n=100, plot_every=1)
    print "... reconstructed"

    # TODO use sklearn to obtain accuracy/precision etc

    # Create Dataset to feed into logistic regression
    # Test set: reconstructed y's become the input. Get the corresponding x's and y's
    dataset01[2] = (theano.shared(reconstructed_y), test_y)

    # Classify the reconstructions TODO
    # score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset01, 100)
    #
    # print 'Score: {}'.format(str(score))
    # print str(rbm)


def associate_data2dataDBN(cache=False):
    print "Testing Associative DBN which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('associative_dbn_test', log=True)


    # Load mnist hand digits, class label is already set to binary
    train, valid, test = m_loader.load_digits(n=[500, 100, 100], digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pre={'binary_label': True})
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
                for n_think in [0, 1, 3, 5, 7, 10]: #1, 3, 5, 7, 10]:
                    # Reconstruct
                    sampled = ass_dbn.recall(test_x, n_recall, n_think)

                    # Sample from top layer to generate data
                    sample_n = 100
                    utils.save_images(sampled, image_name='reconstruced_{}_{}_{}.png'.format(n_ass, n_recall, n_think), shape=(sample_n / 10, 10))

                    dataset01[2] = (theano.shared(sampled), test_y)

                    # Classify the reconstructions TODO
                    # score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset01, 100)
                    #
                    # print 'Score: {}'.format(str(score))
                    # logging.info('{}, {}, {}, {}: {}'.format(cd_type, n_ass, n_recall, n_think, score))


if __name__ == '__main__':
    # associate_data2label()
    associate_data2data(True)
    # associate_data2dataDBN(False)
