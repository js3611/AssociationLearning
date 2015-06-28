__author__ = 'joschlemper'

import theano
import rbm_units


t_float_x = theano.config.floatX

CLASSICAL = 'classical'
NESTEROV = "nesterov"
PERSISTENT = "persistent"


class TrainParam(object):
    def __init__(self,
                 epochs=15,                     # monitor weight
                 batch_size=20,                 # in range [10, 100]
                 learning_rate=0.001,           # use histogram
                 momentum_type=NESTEROV,        # {CLASSICAL, NESTEROV}
                 momentum=0.5,                  # in range [0.5, 0.9]
                 weight_decay=0.001,            # in range [0.0001, 0.01]
                 sparsity_constraint=False,
                 sparsity_target=0.01,          # in range {0.1, 0.01}
                 sparsity_cost=0.01,            # use histogram
                 sparsity_decay=0.9,            # in range [0.9, 0.99]
                 dropout=False,
                 dropout_rate=0.8               # in range [0.5 0.9]
                 ):
        self.epochs = epochs
        self.batch_size = batch_size
        # Weight Update Parameters
        self.learning_rate = learning_rate

        self.momentum_type = momentum_type
        self.momentum = momentum

        self.weight_decay = weight_decay
        self.weight_decay_for_bias = True  # if false, weight decay not applied to biases

        # Sparsity Constraint Parameters
        self.sparsity_constraint = sparsity_constraint
        self.sparsity_target = sparsity_target
        self.sparsity_cost = sparsity_cost
        self.sparsity_decay = sparsity_decay

        self.dropout = dropout
        self.dropout_rate = dropout_rate


    def __str__(self):
        return "batch{}_lr{}_{}{}_wd{}".format(self.batch_size, self.learning_rate, self.momentum_type, self.momentum,
                                               self.weight_decay) + \
               ("_sparsity"
                + "_t" + str(self.sparsity_target)
                + "_c" + str(self.sparsity_cost) +
                "_d" + str(self.sparsity_decay) if self.sparsity_constraint else "") + \
               ("_dropout%.2f" % self.dropout_rate if self.dropout else "")


class RBMConfig(object):
    def __init__(self,
                 cd_type=CLASSICAL,
                 cd_steps=1,
                 associative=False,
                 v_n=10,
                 v_unit=rbm_units.RBMUnit,
                 v2_n=10,
                 v2_unit=rbm_units.RBMUnit,
                 h_n=10,
                 h_unit=rbm_units.RBMUnit,
                 train_params=TrainParam(),
                 progress_logger=None):
        self.cd_type = cd_type
        self.cd_steps = cd_steps
        self.associative = associative

        self.v_n = v_n
        self.v_unit = v_unit

        self.v2_n = v2_n
        self.v2_unit = v2_unit

        self.h_n = h_n
        self.h_unit = h_unit

        self.train_params = train_params
        self.progress_logger = progress_logger

    def __str__(self):
        return '{}{}_{}{}_{}{}'.format(self.cd_type, self.cd_steps, self.v_unit, self.v_n, self.h_unit, self.h_n)
