__author__ = 'joschlemper'

import numpy as np
import theano
import theano.tensor as T
import scipy
from simple_classifiers import SimpleClassifier
from rbm import RBM
from rbm_config import *
from rbm_logger import *
from rbm_units import *
from datastorage import StorageManager
import kanade_loader


def experimentHappyChild():
    # Project set up

    # Get dataset
    #train_x, train_y, test_x, test_y = kanade_loader.load_pairs(e_c, p_config, n=100)

    # Initialise RBM


    # Train RBM


    # Get reconstruction on train image

    # Train classifier on reconstruction


    # Get reconstruction on test image


    # Output number of classes
