__author__ = 'js3611'

from models.rbm import RBM
from models.rbm_config import *
from models.rbm_logger import *
from models import DBN
import kanade_loader as k_loader
from datastorage import StorageManager
# from matplotlib.pyplot import plot, show, ion
# import matplotlib.pyplot as plt
from models.simple_classifiers import SimpleClassifier


def get_rbm_config(shape, n_hidden=500, epochs=10):
    # Initialise RBM
    tr = TrainParam(learning_rate=0.0001,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.0001,
                    sparsity_constraint=True,
                    sparsity_decay=0.9,
                    sparsity_cost=0.01,
                    sparsity_target=0.1,
                    dropout=True,
                    dropout_rate=0.5,
                    batch_size=10,
                    epochs=epochs)

    n_visible = shape * shape
    config = RBMConfig()
    config.v_n = n_visible
    config.h_n = n_hidden
    config.v_unit = rbm_units.GaussianVisibleUnit
    # config.h_unit = rbm_units.ReLUnit
    config.progress_logger = ProgressLogger(img_shape=(shape, shape))
    config.train_params = tr
    return config


def get_dbn_config(shape, data_manager, n_hidden=500, lr=0.01, epochs=10, l=2):
    # Initialise RBM parameters
    base_tr = TrainParam(learning_rate=0.0001,
                         momentum_type=NESTEROV,
                         momentum=0.5,
                         weight_decay=0.0001,
                         sparsity_constraint=True,
                         sparsity_decay=0.9,
                         sparsity_cost=0.01,
                         sparsity_target=0.1,
                         dropout=True,
                         dropout_rate=0.5,
                         batch_size=10,
                         epochs=epochs)

    rest_tr = TrainParam(learning_rate=lr,
                         momentum_type=NESTEROV,
                         momentum=0.5,
                         weight_decay=0.0001,
                         sparsity_constraint=True,
                         sparsity_target=0.1,
                         sparsity_decay=0.9,
                         sparsity_cost=0.01,
                         dropout=True,
                         dropout_rate=0.5,
                         batch_size=10,
                         epochs=epochs, )

    top_tr = TrainParam(learning_rate=lr,
                        momentum_type=NESTEROV,
                        momentum=0.5,
                        weight_decay=0.0001,
                        sparsity_constraint=True,
                        sparsity_target=0.1,
                        sparsity_decay=0.9,
                        sparsity_cost=0.01,
                        dropout=True,
                        dropout_rate=0.5,
                        batch_size=10,
                        epochs=epochs,
                        )

    topology = [(shape ** 2), n_hidden, n_hidden, n_hidden]
    # batch_size = 10
    first_progress_logger = ProgressLogger(img_shape=(shape, shape),monitor_weights=False)
    rest_progress_logger = ProgressLogger(monitor_weights=False)

    first_rbm_config = RBMConfig(train_params=base_tr,
                                 progress_logger=first_progress_logger)
    first_rbm_config.v_unit = rbm_units.GaussianVisibleUnit
    rest_rbm_config = RBMConfig(train_params=rest_tr,
                                progress_logger=rest_progress_logger)
    top_rbm_config = RBMConfig(train_params=top_tr,
                               progress_logger=rest_progress_logger)

    rbm_configs = [first_rbm_config, rest_rbm_config, top_rbm_config]

    topology = topology[:(l+1)]
    rbm_configs = rbm_configs[:l]
    
    config = DBN.DBNConfig(topology=topology,
                           training_parameters=base_tr,
                           rbm_configs=rbm_configs,
                           data_manager=data_manager)

    return config


def noise_classification(project_name = 'NoiseClassification', emotions = {'happy':.9, 'sadness':0.1}):
    # Project set up

    manager = StorageManager(project_name, log=True)

    # Initialise dataset
    dataset = k_loader.load_kanade(set_name='25_25', pre={'scale': True}, emotions=emotions, n=1000)
    tr, vl, te = dataset
    tr_x, tr_y = tr
    te_x, te_y = te

    tr, vl, te= k_loader.load_kanade(shared=True, set_name='25_25', pre={'scale': True}, emotions=['happy','sadness'], n=10000)
    clf_tr_x, clf_tr_y = tr
    clf = SimpleClassifier('knn', clf_tr_x, clf_tr_y)

    emotions = ['sadness']
    noisy_data = []
    noisy_label = []
    noisy_levels = ['', 'noise0.1_', 'noise0.3_','noise0.5_','noise0.7_','noise0.9_']
    for noise_lvl in noisy_levels:
        t, vl, te = k_loader.load_kanade(set_name='{}25_25'.format(noise_lvl), pre={'scale': True}, emotions=emotions, n=1000)
        n_tr_x, n_tr_y = t
        noisy_data.append(n_tr_x)
        noisy_label.append(n_tr_y.eval())

    assess_rbm(clf, noisy_data, noisy_label, noisy_levels, tr_x,'2')
    # assess_dbn(clf, noisy_data, noisy_label, noisy_levels, tr_x, manager,'2')




def assess_rbm(clf, noisy_data, noisy_label, noisy_levels, tr_x,postfix=''):
    f_score = open('report{}.txt'.format(postfix), 'a')
    f_metric = open('metric{}.txt'.format(postfix), 'a')
    # Initialise architecture
    config = get_rbm_config(25, n_hidden=500, epochs=2)
    model = RBM(config)
    pred_table = {}
    for l in xrange(len(noisy_levels)):
        pred_table[l] = []
    for i in xrange(50):
        # Train architecture
        model.train(tr_x)

        j = 0
        for xs, ys in zip(noisy_data, noisy_label):
            recon_xs = model.reconstruct(xs, img_name='test_rbm')
            pred, metric = clf.get_score(recon_xs, ys, True)
            print pred
            print metric
            f_metric.write('{}25_25, Epoch:{}\n'.format(noisy_levels[j], i))
            f_metric.write(metric)
            pred_table[j].append(pred)
            j += 1
    for k in pred_table:
        f_score.write('{}:{}\n'.format(noisy_levels[k], pred_table[k]))
    f_score.close()
    f_metric.close()


def assess_dbn(clf, noisy_data, noisy_label, noisy_levels, tr_x, manager, postfix=''):
    f_score = open('dbn3_report{}.txt'.format(postfix), 'a')
    f_metric = open('dbn3_metric{}.txt'.format(postfix), 'a')
    epochs = 4
    # Initialise architecture
    pred_table = {}
    for l in xrange(len(noisy_levels)*2):
        pred_table[l] = []
    for i in xrange(25):
        # Train architecture
        config = get_dbn_config(25, data_manager=manager, n_hidden=500, epochs=epochs, l=2)
        new_epochs = epochs + i * epochs
        config.rbm_configs[1].train_params.epochs = new_epochs
        model = DBN.DBN(config)
        # model.pretrain(tr_x,
        #                cache=['epoch{}'.format(new_epochs - epochs), False, False],
        #                train_further=[True, True, True], names=['epoch{}'.format(new_epochs)]*3)

        model.pretrain(tr_x,
                       cache=['epoch{}'.format(new_epochs), 'epoch{}'.format(new_epochs), False],
                       train_further=[False, False, True], names=['epoch{}'.format(new_epochs)]*3)

        j = 0
        for xs, ys in zip(noisy_data, noisy_label):
            recon_xs = model.reconstruct(xs, img_name='test_dbn')
            pred, metric = clf.get_score(recon_xs, ys, True)
            print pred
            print metric
            f_metric.write('{}25_25, Epoch:{}\n'.format(noisy_levels[j], i))
            f_metric.write(metric)
            pred_table[j].append(pred)
            j += 1

        model.fine_tune(tr_x, epochs=1)

        for xs, ys in zip(noisy_data, noisy_label):
            recon_xs = model.reconstruct(xs, img_name='test_dbn')
            pred, metric = clf.get_score(recon_xs, ys, True)
            print pred
            print metric
            f_metric.write('[FT] {}25_25, Epoch:{}\n'.format(noisy_levels[j% len(noisy_levels)], i))
            f_metric.write(metric)
            pred_table[j].append(pred)
            j += 1

    for k in pred_table:
        f_score.write('{}:{}\n'.format(noisy_levels[k % len(noisy_levels)], pred_table[k]))
    f_score.close()
    f_metric.close()


if __name__ == '__main__':

    import sys

    if len(sys.argv) > 1:
        ratio_type = float(sys.argv[1])
        print sys.argv
        if ratio_type == 1:
            print '===================SAD50===================='
            noise_classification('Sad50', emotions={'happy':0.5,'sadness':0.5})
        elif ratio_type == 2:
            print '===================SAD75===================='
            noise_classification('Sad75', emotions={'happy':0.75,'sadness':0.25})
        elif ratio_type == 3:
            print '===================SAD90===================='
            noise_classification('Sad90', emotions={'happy':0.9,'sadness':0.1})
    else:
        noise_classification('Sad50',emotions={'happy':0.5,'sadness':0.5})
        noise_classification('Sad25',emotions={'happy':0.75,'sadness':0.25})
        noise_classification('Sad10',emotions={'happy':0.9,'sadness':0.1})

        