__author__ = 'joschlemper'

import m_loader
import kanade_loader as k_loader
import logistic_sgd
import datastorage as store
from simple_classifiers import SimpleClassifier
from rbm import *
from DBN import *
import logging

def find_associative_dbn_hyper_parameters():
    proj_name = 'AssociativeDBN_params'

    # Create project manager, loggers
    manager = store.StorageManager(proj_name)
    f = open('score.txt', 'wr')
    logging.basicConfig(filename=proj_name+'.log', level=logging.INFO)
    logging.info("Starting the project: " + proj_name)

    # Load kanade database
    mapping = {'anger': 'sadness', 'contempt': 'happy', 'disgust': 'sadness', 'fear': 'sadness', 'happy': 'happy', 'sadness': 'sadness', 'surprise': 'happy'}
    train, valid, test = k_loader.load_kanade(n=100, set_name='sharp_equi25_25', pre={'scale2unit': True})
    train_x, train_y = train
    test_x, test_y = test
    train01 = k_loader.sample_image(train_y, mapping=mapping)  # Sample associated image
    train01_x, train01_y = train01 # Sample associated image
    dataset01 = k_loader.load_kanade(n=100, set_name='sharp_equi25_25', emotions=['sadness','happy'], pre={'scale2unit': True})  # Target Image

    # Create Classifier
    clf = SimpleClassifier(classifier='logistic', train_x=dataset01[0][0], train_y=dataset01[0][1])

    # AssociativeRBM Parameter
    n_visible = train_x.get_value().shape[1]
    n_visible2 = n_visible
    sparsity_target_range = [0.01]  # [0.01, 0.001]
    sparsity_cost_range = [0.5]    # histogram mean activities of the hidden units
    sparsity_decay_range = [0.9]
    n_hidden_range = [100, 500]
    cd_types = [CLASSICAL]
    cd_step_range = [1]
    learning_rate_options = [0.005]
    momentum_types = [NESTEROV]
    momentum_range = [0.5]  #[0.1, 0.5, 0.9]
    weight_decay_range = [0.001]# [0.0001, 0.0005, 0.001]
    # sparsity_target_range = [0.1 ** 9, 0.1 ** 7, 0.00001, 0.001, 0.01]
    # sparsity_cost_range = [0.01, 0.1, 0.5]    # histogram mean activities of the hidden units
    # sparsity_decay_range = [0.9, 0.95, 0.99]
    config = RBMConfig()
    config.v_n = n_visible
    config.v2_n = n_visible2
    config.progress_logger = AssociationProgressLogger(img_shape=(25, 25))

    # Iterate through parameters
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

                                            tr = TrainParam(learning_rate=lr,
                                                            momentum_type=mt,
                                                            momentum=m,
                                                            weight_decay=wd,
                                                            sparsity_constraint=False,
                                                            sparsity_target=st,
                                                            sparsity_cost=sc,
                                                            sparsity_decay=sd)

                                            config.h_n = n_hidden
                                            config.associative = False
                                            config.train_params = tr
                                            config.cd_type = cd_type
                                            config.cd_steps = cd_steps

                                            for pen in [100, 250, 500]:
                                                for top in [100, 250, 500]:

                                                    topology = [625, n_hidden, pen, top]
                                                    first_progress_logger = ProgressLogger(img_shape=(25, 25))
                                                    rest_progress_logger = ProgressLogger()

                                                    first_rbm_config = RBMConfig(train_params=tr,
                                                                                 progress_logger=first_progress_logger)
                                                    rbm_config = RBMConfig(train_params=tr,
                                                                                 progress_logger=rest_progress_logger)
                                                    rbm_configs = [first_rbm_config, rbm_config, rbm_config]

                                                    config = DBNConfig(topology=topology,
                                                                       training_parameters=tr,
                                                                       rbm_configs=rbm_configs,
                                                                       data_manager=manager)

                                                    # construct the Deep Belief Network
                                                    dbn = DBN(config)

                                                    if os.path.isdir(os.path.join('data', proj_name, str(counter))):
                                                        logging.info("Skipping {} as it was already sampled".format(counter))
                                                        continue


                                                    converged=False
                                                    min_cost = - float('-inf')
                                                    for j in xrange(0, 5):
                                                        # Train RBM - learn joint distribution
                                                        try:
                                                            curr_cost = dbn.pretrain(train_x)
                                                            if min_cost < np.min(curr_cost):
                                                                # No longer improves
                                                                logging.info('Converged, moving on')
                                                                break
                                                            min_cost = min(np.min(curr_cost), min_cost)

                                                        except Exception as inst:
                                                            logging.info(inst)
                                                            break
                                                        # manager.persist(rbm)

                                                    manager.move_to(str(counter))

                                                    print "... reconstruct via dbn"
                                                    reconstructed_tr = dbn.reconstruct(train_x, k=5, plot_n=100, plot_every=1, img_name="{}_{}_sampled.png".format(counter, j))
                                                    reconstructed_te = dbn.reconstruct(test_x, k=5, plot_n=100, plot_every=1, img_name="{}_{}_sampled.png".format(counter, j))

                                                    # Classify the reconstructions
                                                    score_orig = clf.get_score(reconstructed_te, test_y.eval())
                                                    clf.retrain(reconstructed_tr, train_y.eval())
                                                    score_retrain = clf.get_score(reconstructed_te, test_y.eval())

                                                    out_msg = '{} (orig, retrain):{},{}'.format(dbn, score_orig, score_retrain)
                                                    logging.info(out_msg)
                                                    print out_msg
                                                    f.write(out_msg + '\n')


                                                    logging.info(out_msg)
                                                    print out_msg
                                                    f.write(out_msg + '\n')

                                                    # print str(rbm)

                                                    manager.move_to_project_root()
                                                    counter += 1



    logging.info("End of finding parameters")
    f.close()



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
    dataset = m_loader.load_digits(n=[100, 100, 100])
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
    find_associative_dbn_hyper_parameters()
