import unittest
import datastorage as store
import utils
import numpy as np
import rbm as rbm
import rbm_units
import dbn as DBN
import sklearn
import theano
import theano.tensor as T

import mnist_loader as m_loader


class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_set_up(self):
        self.gaussian_rbm = rbm.GaussianRBM()
        self.assertTrue(self.gaussian_rbm)

    def test_subclass(self):
        self.gaussian_rbm = rbm.GaussianRBM()
        self.assertTrue(self.gaussian_rbm.variance == 1)
        self.assertTrue(self.gaussian_rbm.cd_steps == 1)
        self.assertTrue('gaussian' in str(self.gaussian_rbm))

    def test_inheritance(self):
        class A(object):
            def call_method(self):
                return self.class_name()
            def class_name(self):
                return "A"

        class B(A):
            def class_name(self):
                return "B"

        obj_a = A()
        obj_b = B()

        self.assertTrue(obj_a.call_method() == 'A')
        self.assertTrue(obj_b.call_method() == 'B')

    def test_sk_processing(self):
        x_in = sklearn.preprocessing.scale(np.array([[1.0, 1.0, 1.0]], dtype=float))

        unit = rbm_units.NReLUnit(3)
        x = T.matrix("x")
        y = unit.activate(x)
        f = theano.function([x], [y])
        # print f(x_in)

    def test_axis(self):
        xs = np.array([[1, 2, 3], [4, 5, 6]])
        x = T.matrix('x')
        m1 = T.mean(x)
        m2 = T.mean(x, axis=0)
        m3 = T.mean(x, axis=1)

        f = theano.function([x], [m1, m2, m3])
        v1, v2, v3 = f(xs)

        self.assertTrue(np.all(v1 == 3.5))
        self.assertTrue(np.all(v2 == np.array([2.5, 3.5, 4.5])))
        self.assertTrue(np.all(v3 == np.array([2, 5])))

    def test_full(self):
        train, valid, test = m_loader.load_digits(n=[10000, 0, 100], pre={'scale':True})
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        tr = rbm.TrainParam(learning_rate=0.0001,
                            momentum_type=rbm.CLASSICAL,
                            momentum=0.5,
                            weight_decay=0.01,
                            sparsity_constraint=False,
                            sparsity_target=0.01,
                            sparsity_cost=0.01,
                            sparsity_decay=0.1,
                            epochs=10)

        n_visible = train_x.get_value().shape[1]
        n_hidden = 100

        gaussian_rbm = rbm.GaussianRBM(n_visible,
                                       n_visible,
                                       n_hidden,
                                       associative=False,
                                       cd_type=rbm.CLASSICAL,
                                       cd_steps=1,
                                       visible_unit=rbm_units.GaussianVisibleUnit,
                                       hidden_unit=rbm_units.RBMUnit,
                                       train_parameters=tr,
                                       progress_logger=rbm.ProgressLogger())

        curr_dir = store.move_to('simple_gaussian_rbm_test')
        print "... moved to {}".format(curr_dir)

        # Train RBM
        gaussian_rbm.train(train_x)

        # Test RBM
        # gaussian_rbm.plot_samples(test_x)



if __name__ == '__main__':
    print "Test Utilities"
    unittest.main()