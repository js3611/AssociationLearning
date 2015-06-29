import os

import kanade_loader as loader
import datastorage as store
from models.rbm import RBM
from models.rbm_config import *
from models.rbm_logger import *
from simple_classifiers import SimpleClassifier


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

    data_manager = store.StorageManager('TestRBM')
    # Load Cohn Kanade dataset
    datasets = loader.load_kanade(pre={'scale': True}, n=100, set_name='sharp_equi25_25')
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    # Initilise the RBM and training parameters
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

    n_visible = train_set_x.get_value(borrow=True).shape[1]
    n_hidden = 10

    config = RBMConfig()
    config.v_n = n_visible
    config.h_n = n_hidden
    config.v_unit = rbm_units.GaussianVisibleUnit
    # config.h_unit = rbm_units.ReLUnit
    config.progress_logger = ProgressLogger(img_shape=(25, 25))
    config.train_params = tr
    rbm = RBM(config)
    print "... initialised RBM"

    load = store.retrieve_object(str(rbm))
    if load:
        rbm = load

    for i in xrange(0, 1):
        # Train RBM
        rbm.train(train_set_x)
        data_manager.persist(rbm)

        # Test RBM Reconstruction via Linear Classifier
        clf = SimpleClassifier(classifier='logistic', train_x=train_set_x, train_y=train_set_y)
        recon_te = rbm.reconstruct(test_set_x, k=1, plot_n=100, plot_every=1,img_name='recon_te_{}.png'.format(i))

        print 'Original Score: {}'.format(clf.get_score(test_set_x, test_set_y))
        print 'Recon Score:    {}'.format(clf.get_score(recon_te, test_set_y.eval()))


if __name__ == '__main__':
    test_rbm()
