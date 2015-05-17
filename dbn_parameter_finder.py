__author__ = 'joschlemper'

import mnist_loader as loader
import logistic_sgd
import datastorage as store
from rbm import *
from dbn import *
import logging

logging.basicConfig(filename='trace.log', level=logging.INFO)

def find_hyper_parameters():
    progress_logger = ProgressLogger()
    logging.info("Start Parameter Finder")
    f = open('dbn_score.txt','w')
    manager = store.StorageManager('dbn_params_finder')

    print "Creating Best generative DBN."

    # fixed base train param
    base_tr = TrainParam(learning_rate=0.01,
                    momentum_type=CLASSICAL,
                    momentum=0.5,
                    weight_decay=0.0005,
                    sparsity_constraint=False)

    # Load mnist hand digits, class label is already set to binary
    dataset = loader.load_digits(n=[100, 100, 100])
    train_x, train_y = dataset[0]


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

    n_hidden_range = [500]
    cd_types = [CLASSICAL, PERSISTENT]
    cd_step_range = [1, 3]
    learning_rate_options = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.003, 0.005]#[0.0001, 0.001, 0.01, 0.1] # Histogram
    momentum_types = [CLASSICAL, NESTEROV]
    momentum_range = [0.3, 0.5]  #[0.1, 0.5, 0.9]
    weight_decay_range = [0.001, 0.0001, 0.01]# [0.0001, 0.0005, 0.001]
    # sparsity_target_range = [0.1 ** 9, 0.1 ** 7, 0.00001, 0.001, 0.01]
    # sparsity_cost_range = [0.01, 0.1, 0.5]    # histogram mean activities of the hidden units
    # sparsity_decay_range = [0.9, 0.95, 0.99]

    possibilities = reduce(lambda x, y: x*y,
                           map(lambda x: len(x), [n_hidden_range, cd_types, cd_step_range, learning_rate_options,
                                                  momentum_types, momentum_range, weight_decay_range,
                                                  sparsity_target_range, sparsity_cost_range, sparsity_decay_range]), 1)

    logging.info(str(possibilities) + " parameter sets to explore")

    # Keep track of the best one
    classical_max_cost = - float('-inf')
    persistent_max_cost = - float('-inf')
    classical_max_name = ''
    persistent_max_name = ''
    counter = 1

    for sd in sparsity_decay_range:
        for sc in sparsity_cost_range:
            for st in sparsity_target_range:
                for cd_steps in cd_step_range:
                    for mt in momentum_types:
                        for m in momentum_range:
                            for wd in weight_decay_range:
                                for cd_type in cd_types:
                                    for n_hidden in n_hidden_range:
                                        for lr in learning_rate_options:
                                            # Initialise the RBM and training parameters
                                            logging.info("Search Progress: {} / {}".format(str(counter), possibilities))
                                            counter+=1

                                            tr = TrainParam(learning_rate=lr,
                                                            momentum_type=mt,
                                                            momentum=m,
                                                            weight_decay=wd,
                                                            sparsity_constraint=False)

                                            for pen in [100, 250, 500]:
                                                for top in [100, 250, 500, 750]:

                                                    topology = [784, 500, pen, top]
                                                    n_layers = len(topology)
                                                    trlist = [base_tr] + [tr, tr]
                                                    dbn = DBN(topology=topology, tr=trlist, data_manager=manager)
                                                    dbn.pretrain(train_x)

                                                    sample_n = 100
                                                    sampled = dbn.sample(sample_n, 2)

                                                    utils.save_digits(sampled, shape=(sample_n / 10, 10), image_name=str(counter) + '_' + str(dbn) + ".png")


                                                    dataset[2] = (theano.shared(sampled), dataset[2][1])
                                                    score = logistic_sgd.sgd_optimization_mnist(0.13, 100, dataset, 10)

                                                    logging.info(str(dbn) + " : " + str(score))
                                                    f.write(str(dbn) + ':' + str(score) + '\n')

                                                    print str(dbn)

    logging.info("End of finding parameters")
    f.close()

if __name__ == '__main__':
    find_hyper_parameters()
