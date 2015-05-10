import theano
import theano.tensor as T
import numpy as np
import logistic_sgd
import rbm as RBM
import utils
import mnist_loader as loader
import datastorage as store


def associate_data2label(cache=False):
    print "Testing ClassRBM with generative target (i.e. AssociativeRBM with picture-label association)"

    # Load mnist hand digits, class label is already set to binary
    train, valid, test = loader.load_digits(n=[500, 100, 100], pre={'label_vector': True})
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
    rbm.reconstruct(test_x.get_value(borrow=True), 10)
    rbm.associative = True

    # Classification test - Reconstruct y through x
    output_probability = rbm.reconstruct_association(test_x, None)
    sol = np.array([np.argmax(lab) for lab in test_y.eval()])
    guess = np.array([np.argmax(lab) for lab in output_probability])
    diff = np.count_nonzero(sol == guess)

    print "Classification Rate: {}".format((diff / float(test_y.eval().shape[0])))


def associate_data2data(cache=False):
    print "Testing Associative RBM which tries to learn even-oddness of numbers"

    # Load mnist hand digits, class label is already set to binary
    train, valid, test = loader.load_digits(n=[50000, 100, 10000], digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = loader.sample_image(train_y)

    dataset01 = loader.load_digits(n=[50000, 100, 10000], digits=[0, 1])


    # Initialise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.0001,
                        momentum_type=RBM.CLASSICAL,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=False,
                        sparsity_target=0.01,
                        sparsity_cost=0.5,
                        sparsity_decay=0.9,
                        epochs=20)

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
                  cd_steps=1,
                  train_parameters=tr,
                  progress_logger=RBM.AssociationProgressLogger())

    store.move_to('AssociationTest')

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded and cache:
        rbm = loaded
        print "... loaded precomputed rbm"
    else:
        # Train RBM - learn joint distribution
        rbm.train(train_x, train_x01)
        rbm.save()

    print "... reconstruction of associated images"
    reconstructed_y = rbm.reconstruct_association(test_x, None, 5, 0.01, sample_size=200)
    print "... reconstructed"

    # TODO use sklearn to obtain accuracy/precision etc

    # Create Dataset to feed into logistic regression
    # Test set: reconstructed y's become the input. Get the corresponding x's and y's
    dataset01[2] = (theano.shared(reconstructed_y), test_y)

    # Classify the reconstructions
    score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset01, 100)

    print 'Score: {}'.format(str(score))
    print str(rbm)


def associate_data2dataDBN(cache=False):
    pass
    # # Load data
    # train, valid, test = loader.load_digits(n=[500, 100, 100], digits=[0, 1, 2, 3, 4, 5])
    # train_x, train_y = train
    #
    # # Initialise RBM parameters
    # tr = TrainParam(learning_rate=0.01,
    #                 momentum_type='nesterov',
    #                 momentum=0.5,
    #                 weight_decay=0.01,
    #                 sparsity_constraint=True,
    #                 sparsity_target=0.01,
    #                 sparsity_cost=0.01,
    #                 sparsity_decay=0.1,
    #                 epochs=15)
    #
    # # Layer 1
    # # Layer 2
    # # Layer 3
    # topology = [784, 100, 100, 100]
    # batch_size = 10
    #
    # # construct the Deep Belief Network
    # dbn = DBN(topology=topology, n_outs=10, out_dir='zero_one_learner', tr=tr)
    # print "... initialised dbn"
    #
    # store.move_to(dbn.out_dir)
    # print "... moved to {}".format(os.getcwd())
    #
    # print '... pre-training the model'
    # start_time = time.clock()
    #
    # # dbn.pretrain(train_x, cache=True)
    # dbn.pretrain(train_x, cache=False)
    #
    # end_time = time.clock()
    # print >> sys.stderr, ('The pretraining code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))
    #
    # store.move_to(dbn.out_dir)
    # store.store_object(dbn)
    #
    # print "... moved to {}".format(os.getcwd())
    #
    # # Sample from top layer to generate data
    # sample_n = 100
    # sampled = dbn.sample(sample_n, 10)
    #
    # save_digits(sampled, shape=(sample_n / 10, 10))

if __name__ == '__main__':
    # associate_data2label()
    associate_data2data(True)
