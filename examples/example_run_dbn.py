import os

import datastorage as store
import mnist_loader as m_loader
import kanade_loader as k_loader
from models.rbm_config import *
from models.rbm_logger import *
from models.DBN import DBN, DBNConfig

theano.config.optimizer = 'None'


def test_generative_dbn():
    manager = store.StorageManager('fine_tune')
    shape = 28
    train_x = get_data(shape)

    # Initialise RBM parameters
    dbn = get_dbn_model(manager, shape)

    print "... initialised dbn"

    print '... pre-training the model'
    start_time = time.clock()

    dbn.pretrain(train_x, cache=[True, True], train_further=[True, True])
    # dbn.pretrain(train_x, cache=False)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    # Sample from top layer to generate data
    sample_n = 1000
    sampled = dbn.sample(sample_n, k=100, rand_type='noisy_mean')
    k_loader.save_faces(sampled, tile=(sample_n / 10, 10), img_name="sampled.png", img_shape=(shape, shape))
    dbn.reconstruct(train_x, k=1, plot_every=1, plot_n=100, img_name='dbn_recon')

    for i in xrange(0,1):
        dbn.fine_tune(train_x, epochs=1)
        sampled = dbn.sample(sample_n, k=100, rand_type='noisy_mean')
        k_loader.save_faces(sampled, tile=(sample_n / 10, 10), img_name=("sampled_fine_tuned%d.png" % i), img_shape=(shape, shape))
        dbn.reconstruct(train_x, k=1, plot_every=1, plot_n=100, img_name=('dbn_recon_fine_tune%d' % i))


def get_data(shape):
    dataset_name = 'sharp_equi{}_{}'.format(shape, shape)
    # Load data
    train, valid, test = m_loader.load_digits(n=[1000, 10, 100])
    # train, valid, test = k_loader.load_kanade(set_name=dataset_name, pre={'scale': True})
    train_x, train_y = train
    return train_x


def get_dbn_model(manager, shape):
    # Layer 1
    tr = TrainParam(learning_rate=0.001,
                    momentum_type=NESTEROV,
                    momentum=0.5,
                    weight_decay=0.0001,
                    sparsity_constraint=False,
                    sparsity_target=0.1,
                    sparsity_decay=0.9,
                    sparsity_cost=0.1,
                    dropout=True,
                    dropout_rate=0.5,
                    epochs=10)

    first_progress_logger = ProgressLogger(img_shape=(shape, shape))
    first_rbm_config = RBMConfig(train_params=tr, progress_logger=first_progress_logger)
    # first_rbm_config.v_unit = rbm_units.GaussianVisibleUnit


    # Layer Mid
    # --
    rest_progress_logger = ProgressLogger()


    # Layer Top
    top_tr = TrainParam(learning_rate=0.0001,
                        momentum_type=NESTEROV,
                        momentum=0,
                        weight_decay=0,
                        sparsity_constraint=False,
                        sparsity_target=0.1,
                        sparsity_cost=0.1,
                        sparsity_decay=0.9,
                        batch_size=10,
                        epochs=10)

    rbm_config = RBMConfig(train_params=top_tr, progress_logger=rest_progress_logger)


    # DBN Config
    topology = [shape ** 2, 250, 100]
    rbm_configs = [first_rbm_config, rbm_config, rbm_config]
    config = DBNConfig(topology=topology,
                       training_parameters=tr,
                       rbm_configs=rbm_configs,
                       data_manager=manager)
    # construct the Deep Belief Network
    dbn = DBN(config)
    return dbn


if __name__ == '__main__':
    test_generative_dbn()
