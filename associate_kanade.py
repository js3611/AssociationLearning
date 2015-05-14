__author__ = 'joschlemper'

import rbm as RBM
import kanade_loader as loader
import datastorage as store




def train_kanade():
    print "Testing RBM"

    data_manager = store.StorageManager('Kanade/SimpleRBMTest')

    # Load mnist hand digits
    datasets = loader.load_kanade(n=500, pre={'scale2unit': True})
    train_x, train_y = datasets

    sparsity_constraint = False
    # Initialise the RBM and training parameters
    tr = RBM.TrainParam(learning_rate=0.1,
                        momentum_type=RBM.NESTEROV,
                        momentum=0.5,
                        weight_decay=0.0001,
                        sparsity_constraint=sparsity_constraint,
                        sparsity_target=0.001,
                        sparsity_cost=1,
                        sparsity_decay=0.9,
                        epochs=100)

    n_visible = train_x.get_value().shape[1]
    n_hidden = 100

    rbm = RBM.RBM(n_visible,
                  n_visible,
                  n_hidden,
                  associative=False,
                  cd_type=RBM.CLASSICAL,
                  cd_steps=1,
                  train_parameters=tr,
                  progress_logger=RBM.ProgressLogger(img_shape=(25, 25)))

    print "... initialised RBM"

    if sparsity_constraint:
        rbm.get_initial_mean_activity(train_x)


    # adjust learning rate
    rbm.pretrain_lr(train_x)
    # rbm.pretrain_mean_activity_h(train_x)

    # Train RBM
    rbm.train(train_x)

    # Test RBM
    rbm.reconstruct(train_x, k=10, plot_n=10, plot_every=1)

    # Store Parameters
    data_manager.persist(rbm)



if __name__ == '__main__':
    train_kanade()