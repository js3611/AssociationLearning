import numpy as np
import theano
import theano.tensor as T

import kanade_loader as loader
import m_loader
import datastorage as store
from rbm import RBM
from rbm_config import *
from rbm_logger import *
from simple_classifiers import SimpleClassifier

import os
import time

try:
    import PIL.Image as Image
except ImportError:
    import Image

CLASSICAL = 'classical'
NESTEROV = "nesterov"
PERSISTENT = "persistent"

# Macro
t_float_x = theano.config.floatX

# Path
root_dir = os.getcwd()
data_dir = "/".join([root_dir, "data"])

# compute_test_value is 'off' by default, meaning this feature is inactive
# theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature
# theano.config.optimizer='None'
# theano.config.exception_verbosity='high'


def test_rbm():
    print "Testing RBM"

    data_manager = store.StorageManager('FaceRecon')
    # Load mnist hand digits
    # datasets = m_loader.load_digits(shared=False)
    datasets = loader.load_kanade(shared=False, pre={'scale': True}, n=10000, set_name='sharp_equi25_25')
    # datasets = loader.load_kanade(pre={'scale2unit': True})
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initilise the RBM and training parameters
    # tr = TrainParam(learning_rate=0.001,
    tr = TrainParam(learning_rate=0.0001,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.0001,
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_cost=0.1,
                    sparsity_decay=0.9,
                    dropout=True,
                    dropout_rate=0.5,
                    batch_size=10,
                    epochs=10)

    n_visible = train_set_x.shape[1]
    # n_visible = train_set_x.get_value(borrow=True).shape[1]
    n_hidden = 100

    config = RBMConfig()
    config.v_n = n_visible
    config.h_n = n_hidden
    config.v_unit = rbm_units.GaussianVisibleUnit
    # config.h_unit = rbm_units.ReLUnit
    config.progress_logger = ProgressLogger(img_shape=(25, 25))
    # config.progress_logger = ProgressLogger(img_shape=(50, 50))
    config.train_params = tr
    rbm = RBM(config)
    print "... initialised RBM"

    # adjust learning rate
    # rbm.pretrain_lr(train_set_x)

    # rbm.pretrain_mean_activity_h(train_set_x)

    # rbm.get_initial_mean_activity(train_set_x)

    # Pre-training
    # rbm.set_initial_visible_bias(train_set_x)

    # if tr.sparsity_constraint:
    #     rbm.set_initial_hidden_bias()
    #     rbm.set_hidden_mean_activity(train_set_x)

    load = store.retrieve_object(str(rbm))
    if load:
        rbm = load

    for i in xrange(0, 1):
        # Train RBM
        # f = theano.shared(train_set_x[i * 100 : (i+1) * 100].astype(t_float_x), name='data')
        # rbm.train(f, ctr=i)
        # rbm.train(theano.shared(train_set_x.astype(t_float_x)))
        # rbm.train(train_set_x)
        data_manager.persist(rbm)

        f = theano.shared(train_set_x[0:30].astype(t_float_x), name='data')
        recon_te = rbm.reconstruct(f, k=15, plot_n=100, plot_every=1, img_name='recon_te_{}'.format(i))
        # recon_te = rbm.mean_field_inference(f, k=100, plot_n=100, plot_every=1, img_name='mf_te_{}'.format(i))

        # Test RBM Reconstruction via Linear Classifier
        # clf = SimpleClassifier(classifier='logistic', train_x=train_set_x, train_y=train_set_y)
        # recon_tr = rbm.reconstruct(train_set_x, k=100, plot_n=100, plot_every=1,img_name='recon_tr_{}.png'.format(i))
        # recon_te = rbm.reconstruct(test_set_x, k=1, plot_n=100, plot_every=1,img_name='recon_te_{}.png'.format(i))

        # assert len(recon_tr) == len(train_set_x.get_value(borrow=True))
        #
        # print 'Original Score: {}'.format(clf.get_score(test_set_x, test_set_y))
        # print 'Recon Score:    {}'.format(clf.get_score(recon_te, test_set_y.eval()))
        # clf.retrain(recon_tr, train_set_y.eval())
        # print 'Recon Score (retrain): {}'.format(clf.get_score(recon_te, test_set_y.eval()))
        #


if __name__ == '__main__':
    test_rbm()
