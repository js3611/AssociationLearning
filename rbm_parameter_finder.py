__author__ = 'joschlemper'

from rbm import *


def find_hyper_parameters():

    # Load mnist hand digits
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # 2. for loop each optimal parameter
    # learning_rate_options = [0.1]  # Histogram
    # momentum_types = [NESTEROV]
    # momentum_range = [0.5]
    # weight_decay_range = [0.001]
    sparsity_target_range = [0.01, 0.001]
    sparsity_cost_range = [0.5]    # histogram mean activities of the hidden units
    sparsity_decay_range = [0.9]
    #
    # cd_types = [CLASSICAL, PERSISTENT]
    # cd_step_range = [1]
    # n_hidden_range = [500, 1000]

    n_hidden_range = [10, 100, 500, 750, 1000]
    cd_types = [CLASSICAL, PERSISTENT]
    cd_step_range = [1, 5, 10]
    learning_rate_options = [0.0001, 0.001, 0.01, 0.1] # Histogram
    momentum_types = [CLASSICAL, NESTEROV]
    momentum_range = [0.1, 0.5, 0.9]
    weight_decay_range = [0.0001, 0.0005, 0.001]
    # sparsity_target_range = [0.1 ** 9, 0.1 ** 7, 0.00001, 0.001, 0.01]
    # sparsity_cost_range = [0.01, 0.1, 0.5]    # histogram mean activities of the hidden units
    # sparsity_decay_range = [0.9, 0.95, 0.99]


    # Keep track of the best one
    classical_max_cost = - float('-inf')
    persistent_max_cost = - float('-inf')
    classical_max_name = ''
    persistent_max_name = ''

    n_visible = train_set_x.get_value().shape[1]
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
                                            
                                            tr = TrainParam(learning_rate=lr,
                                                            momentum_type=mt,
                                                            momentum=m,
                                                            weight_decay=wd,
                                                            sparsity_constraint=True,
                                                            sparsity_target=st,
                                                            sparsity_cost=sc,
                                                            sparsity_decay=sd,
                                                            plot_during_training=True)

                                            rbm = RBM(n_visible,
                                                      n_hidden,
                                                      cd_type=cd_type,
                                                      cd_steps=cd_steps,
                                                      train_parameters=tr)

                                            if os.path.isdir("data/"+str(rbm)):
                                                print "Skipping " + str(rbm) + " as it was already sampled"
                                                continue                                    

                                            # Train RBM
                                            cost = rbm.train(train_set_x)
                                            if cd_type is PERSISTENT:
                                                if cost > persistent_max_cost:
                                                    persistent_max_cost = cost
                                                    persistent_max_name = str(rbm)
                                            else:
                                                if cost > classical_max_cost:
                                                    classical_max_cost = cost
                                                    classical_max_name = str(rbm)

                                            # Test RBM
                                            rbm.plot_samples(test_set_x)

                                            # Store Parameters
                                            rbm.save()

    print "Best RBM from classical is: " + classical_max_name
    print "Best RBM from persistent is: " + persistent_max_name

if __name__ == '__main__':
    find_hyper_parameters()
