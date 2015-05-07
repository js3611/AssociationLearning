import os
import sys
import time

import numpy as np
import datastorage as store

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from RBM2 import RBM
from RBM2 import TrainParam
from DBN import DBN

try:
    import PIL.Image as Image
except ImportError:
    import Image

from utils import tile_raster_images
from utils import load_data
from utils import save_digits


def test_DBN(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01, k=1, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=10, output_folder='dbn_plots'):

    # Load data
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(topology=[784, 10, 10], n_outs=10)

    # Change directory
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    print "... moved to {}".format(os.getcwd())

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        plotting_time = 0.
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)

            #insert raster_image here
            plotting_start = time.clock()
            dim_arr = [28, 30, 30]
            image = Image.fromarray(
                tile_raster_images(
                    X=dbn.rbm_layers[i].W.get_value(borrow=True).T,
                    img_shape=(dim_arr[i], dim_arr[i]),
                    tile_shape=(10, 10),
                    tile_spacing=(1, 1)
                )
            )
            image.save('l%i_filters_at_epoch_%i.png' % (i, epoch))
            plotting_stop = time.clock()
            plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()
    # end-snippet-2
    # subtracted plotting time
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time - plotting_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


def test_generative_dbn():

    # Load data
    datasets = load_data('mnist.pkl.gz')
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Initialise RBM parameters
    tr = TrainParam(learning_rate=0.01,
                    momentum_type='nesterov',
                    momentum=0.5,
                    weight_decay=0.01,
                    sparsity_constraint=True,
                    sparsity_target=0.01,
                    sparsity_cost=0.01,
                    sparsity_decay=0.1,
                    plot_during_training=True,
                    epochs=2)

    # Layer 1
    # Layer 2
    # Layer 3
    topology = [784, 105, 105]
    batch_size = 10

    # construct the Deep Belief Network
    dbn = DBN(topology=topology, n_outs=10, out_dir='dbn_test', tr=tr)
    print "... initialised dbn"

    store.move_to(dbn.out_dir)
    print "... moved to {}".format(os.getcwd())

    print '... pre-training the model'
    start_time = time.clock()

    # dbn.pretrain(train_set_x, cache=True)
    dbn.pretrain(train_set_x, cache=False)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    store.move_to(dbn.out_dir)
    store.store_object(dbn)

    print "... moved to {}".format(os.getcwd())

    # Sample from top layer to generate data
    sampled = dbn.sample(10, 1000)

    save_digits(sampled)


    # end-snippet-2
    # subtracted plotting time

if __name__ == '__main__':
#     test_DBN(finetune_lr=0.1, pretraining_epochs=30,
#              pretrain_lr=0.01, k=1, training_epochs=100,
#              dataset='mnist.pkl.gz', batch_size=20)

    test_generative_dbn()
