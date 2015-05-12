import numpy as np
import theano
import theano.tensor as T
from activationFunction import *
from theano.tensor.shared_randomstreams import RandomStreams
import utils
import mnist_loader as loader
import datastorage as store
import rbm as RBM
import sklearn

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

    data_manager = store.StorageManager('SimpleRBMTest')

    # Load mnist hand digits
    datasets = loader.load_digits(n=[10000, 0, 100], digits=[0,1])
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initilise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.01,
                        momentum_type=CLASSICAL,
                        momentum=0.1,
                        weight_decay=0.0005,
                        sparsity_constraint=True,
                        sparsity_target=0.01,
                        sparsity_cost=0.5,
                        sparsity_decay=0.9)

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 500

    rbm = RBM.RBM(n_visible,
                  n_visible,
                  n_hidden,
                  associative=False,
                  cd_type=PERSISTENT,
                  cd_steps=1,
                  v_activation_fn=log_sig,
                  h_activation_fn=log_sig,
                  train_parameters=tr,
                  progress_logger=RBM.ProgressLogger())

    print "... initialised RBM"


    curr_dir = store.move_to('simple_rbm_test')
    print "... moved to {}".format(curr_dir)

    # Train RBM
    rbm.train(train_set_x)

    # Test RBM
    rbm.reconstruct(test_set_x, k=2, plot_n=500, plot_every=1)

    # Store Parameters
    rbm.save()

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded:
        print "... loaded trained RBM"

    # Move back to root
    os.chdir(root_dir)
    print "... moved to " + root_dir


def test_rbm():
    print "Testing RBM"

    data_manager = store.StorageManager('SimpleRBMTest')

    # Load mnist hand digits
    datasets = loader.load_digits(n=[5000, 0, 100])
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initilise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0,
                        momentum_type=CLASSICAL,
                        momentum=0.5,
                        weight_decay=0.001,
                        sparsity_constraint=False,
                        sparsity_target=0.01,
                        sparsity_cost=0.5,
                        sparsity_decay=0.9,
                        epochs=20)

    n_visible = train_set_x.get_value().shape[1]
    n_hidden = 1000

    rbm = RBM.RBM(n_visible,
                  n_visible,
                  n_hidden,
                  associative=False,
                  cd_type=CLASSICAL,
                  cd_steps=1,
                  v_activation_fn=log_sig,
                  h_activation_fn=log_sig,
                  train_parameters=tr,
                  progress_logger=RBM.ProgressLogger())

    print "... initialised RBM"

    # adjust learning rate
    rbm.pretrain_lr(train_set_x)

    # Train RBM
    rbm.train(train_set_x)


    rbm.W
    # Test RBM
    rbm.reconstruct(test_set_x, k=1, plot_n=500, plot_every=1)

    # Store Parameters
    rbm.save()

    # Load RBM (test)
    loaded = store.retrieve_object(str(rbm))
    if loaded:
        print "... loaded trained RBM"

    # Move back to root
    os.chdir(root_dir)
    print "... moved to " + root_dir


if __name__ == '__main__':
    test_rbm()
