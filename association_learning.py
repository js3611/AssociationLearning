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
# import matplotlib.pyplot as plt
from simple_classifiers import SimpleClassifier

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

    store.move_to('label_test/' + str(rbm))

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


def associate_data2data(cache=False, train_further=True):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"
    # project set-up
    data_manager = store.StorageManager('EvenOdd', log=True)
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
    #                 sparsity_target=0.1 ** 9,
    #                 sparsity_cost=0.9,
    #                 sparsity_decay=0.99,
    #                 epochs=50)

    tr = TrainParam(learning_rate=0.001,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.0001,
                    sparsity_constraint=True,
                    sparsity_target=0.1,
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
    n_hidden = min(1000, int((- np.log2(0.1)) * train_n / 10))
    print "number of hidden nodes: %d" % n_hidden
    # n_hidden = 100

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
    for i in xrange(0, 5):
        # Train RBM
        if not loaded or train_further:
            rbm.train(tr_concat_x)

        # Save RBM
        data_manager.persist(rbm)

        # Reconstruct using RBM
        y = rbm.np_rand.binomial(1, 0.0, size=(test_n, 784)).astype(t_float_x)

        recon_x = rbm.mean_field_inference_opt(te_x,
                                               y,
                                               # te_x01,
                                               sample=False,
                                               k=10,
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
        pred = clf.classify(recon_x)

        error = np.sum(orig != pred) * 1. / len(orig)
        print error
        errors.append(error)

    # plt.plot(errors)
    # plt.show()
    print errors

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


if __name__ == '__main__':
    # associate_data2label()
    associate_data2data(True, True)
    # associate_data2dataDBN(False)
