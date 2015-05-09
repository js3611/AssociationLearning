__author__ = 'joschlemper'

import unittest

import numpy as np
import scipy as sp
import theano
import mnist_loader as loader
import utils


class MyTestCase(unittest.TestCase):
    def test_load_raw(self):
        dataset = loader.load_digits(shared=False)
        self.assertTrue(len(dataset) == 3)

        train, valid, test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        self.assertTrue(len(train_x[0]) == 784 and
                        len(valid_x[0]) == len(train_x[0]) and
                        len(test_x[0]) == len(train_x[0]))
        self.assertTrue(len(train_x) == len(train_y) and len(train_y) == 50000)
        self.assertTrue(len(valid_x) == len(valid_y) and len(valid_y) == 10000)
        self.assertTrue(len(test_x) == len(test_y) and len(test_y) == 10000)

    def test_load_individual_digits(self):
        chosen_digits = [0, 1]
        dataset = loader.load_digits(shared=False, digits=chosen_digits)

        self.assertTrue(len(dataset) == 3)

        train, valid, test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        self.assertTrue(len(train_x) == len(train_y))
        self.assertTrue(len(valid_x) == len(valid_y))
        self.assertTrue(len(test_x) == len(test_y))
        self.assertTrue((np.unique(train_y) == np.array(chosen_digits)).all())
        self.assertTrue((np.unique(valid_y) == np.array(chosen_digits)).all())
        self.assertTrue((np.unique(test_y) == np.array(chosen_digits)).all())

        utils.save_digits(train_x[0:100], 'test_image/zero_and_one_train.png')
        utils.save_digits(valid_x[0:100], 'test_image/zero_and_one_valid.png')
        utils.save_digits(test_x[0:100], 'test_image/zero_and_one_test.png')

    def test_load_fixed_number(self):
        # [train_n, valid_n, test_n]
        dataset = loader.load_digits(shared=False, n=[500, 400, 300])

        self.assertTrue(len(dataset) == 3)

        train, valid, test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        # Statistics
        # print sp.stats.itemfreq(train_y)

        self.assertTrue(len(train_x[0]) == 784 and
                        len(valid_x[0]) == len(train_x[0]) and
                        len(test_x[0]) == len(train_x[0]))
        self.assertTrue(len(train_x) == len(train_y) and len(train_y) == 500)
        self.assertTrue(len(valid_x) == len(valid_y) and len(valid_y) == 400)
        self.assertTrue(len(test_x) == len(test_y) and len(test_y) == 300)

    def test_shared(self):
        dataset = loader.load_digits(n=[10, 10, 10])
        train, valid, test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        # print type(train_x)
        # print type(train_y)
        self.assertTrue(isinstance(train_x, theano.tensor.sharedvar.TensorSharedVariable))
        self.assertTrue(isinstance(valid_x, theano.tensor.sharedvar.TensorSharedVariable))
        self.assertTrue(isinstance(test_x, theano.tensor.sharedvar.TensorSharedVariable))

    def test_load_preprocessed(self):
        dataset = loader.load_digits(shared=False, pre={'threshold':0.5}, n=[10, 10, 10])
        for (x, y) in dataset:
            # print sp.stats.itemfreq(x)
            self.assertTrue(np.all((np.unique(x) == np.array([0, 1]))))

    def test_load_vectorised(self):
        dataset = loader.load_digits(shared=False, pre={'label_vector':True}, n=[10, 10, 10])
        train, valid, test = dataset
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        self.assertTrue(train_y.shape[1] == 10)
        self.assertTrue(valid_y.shape[1] == 10)
        self.assertTrue(test_y.shape[1] == 10)

    def test_sample_image(self):
        train, valid, test = loader.load_digits(digits=[2, 3], n=[100, 0, 0], pre={'binary_label':True})
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        train_x01 = loader.sample_image(train_y, shared=False)
        print train_y
        utils.save_digits(train_x01, 'test_image/sampled_img.png')

if __name__ == '__main__':
    unittest.main()
