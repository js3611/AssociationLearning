__author__ = 'joschlemper'

import mnist_loader as loader
import logistic_sgd
import datastorage as store
from rbm import *
import logging

logging.basicConfig(filename='trace.log', level=logging.INFO)

def find_hyper_parameters():
    logging.info("Start!")

    print "Testing Associative RBM which tries to learn even-oddness of numbers"

    # Load mnist hand digits, class label is already set to binary
    train, valid, test = loader.load_digits(n=[1000, 100, 100], digits=[0,1], pre={'binary_label': True})
    train_x, train_y = train
    test_x, test_y = test
    train_x01 = loader.sample_image(train_y)

    dataset01 = loader.load_digits(n=[1000, 100, 100], digits=[0, 1])

    n_visible = train_x.get_value().shape[1]
    n_visible2 = n_visible


    # 2. for loop each optimal parameter
    # learning_rate_options = [0.1]  # Histogram
    # momentum_types = [NESTEROV]
    # momentum_range = [0.5]
    # weight_decay_range = [0.001]
    sparsity_target_range = [0.01]  # [0.01, 0.001]
    sparsity_cost_range = [0.5]    # histogram mean activities of the hidden units
    sparsity_decay_range = [0.9]
    #
    # cd_types = [CLASSICAL, PERSISTENT]
    # cd_step_range = [1]
    # n_hidden_range = [500, 1000]

    n_hidden_range = [10, 50, 100, 300, 500, 750, 1000]
    cd_types = [CLASSICAL, PERSISTENT]
    cd_step_range = [1, 3]
    learning_rate_options = [0.005, 0.01, 0.05, 0.1, 0.5]#[0.0001, 0.001, 0.01, 0.1] # Histogram
    momentum_types = [CLASSICAL, NESTEROV]
    momentum_range = [0.5]  #[0.1, 0.5, 0.9]
    weight_decay_range = [0.001, 0.0001, 0.01]# [0.0001, 0.0005, 0.001]
    # sparsity_target_range = [0.1 ** 9, 0.1 ** 7, 0.00001, 0.001, 0.01]
    # sparsity_cost_range = [0.01, 0.1, 0.5]    # histogram mean activities of the hidden units
    # sparsity_decay_range = [0.9, 0.95, 0.99]


    # Keep track of the best one
    classical_max_cost = - float('-inf')
    persistent_max_cost = - float('-inf')
    classical_max_name = ''
    persistent_max_name = ''

    for sd in sparsity_decay_range:
        for sc in sparsity_cost_range:
            for st in sparsity_target_range:
                for cd_steps in cd_step_range:
                    for mt in momentum_types:
                        for m in momentum_range:
                            for wd in weight_decay_range:
                                for lr in learning_rate_options:
                                    for cd_type in cd_types:
                                        for n_hidden in n_hidden_range:
                                            # Initialise the RBM and training parameters

                                            tr = TrainParam(learning_rate=lr,
                                                            momentum_type=mt,
                                                            momentum=m,
                                                            weight_decay=wd,
                                                            sparsity_constraint=False,
                                                            sparsity_target=st,
                                                            sparsity_cost=sc,
                                                            sparsity_decay=sd,
                                                            plot_during_training=True)

                                            rbm = RBM(n_visible,
                                                      n_visible2,
                                                      n_hidden,
                                                      associative=True,
                                                      cd_type=cd_type,
                                                      cd_steps=cd_steps,
                                                      train_parameters=tr)

                                            if os.path.isdir("data/even_odd/"+str(rbm)):
                                                print "Skipping " + str(rbm) + " as it was already sampled"
                                                continue                                    

                                            store.move_to('even_odd/' + str(rbm))

                                            # Train RBM - learn joint distribution
                                            rbm.train(train_x, train_x01)
                                            rbm.save()

                                            print "... reconstruction of associated images"
                                            reconstructed_y = rbm.reconstruct_association(test_x, None, 2, 0.01, sample_size=100)
                                            print "... reconstructed"

                                            # Create Dataset to feed into logistic regression
                                            # Test set: reconstructed y's become the input. Get the corresponding x's and y's

                                            dataset01[2] = (theano.shared(reconstructed_y), test_y)

                                            # Classify the reconstructions
                                            score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset01, 10)

                                            store.move_to_root()

                                            logging.info(str(rbm) + "\n Rate: " + str(score))
                                            print str(rbm)

    logging.info("end!")

if __name__ == '__main__':
    find_hyper_parameters()
