import unittest
import datastorage as store
import utils
import numpy as np
import rbm as rbm
import rbm_config
import rbm_logger
import rbm_units
import sklearn
import theano
import theano.tensor as T
import scipy.stats as ss
import matplotlib.pyplot as plt

import m_loader
import kanade_loader as k_loader


class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_set_up(self):
        tr = rbm_config.TrainParam(learning_rate=0.05,
                    momentum_type=rbm_config.CLASSICAL,
                    momentum=0.5,
                    weight_decay=0,
                    sparsity_constraint=False,
                    sparsity_target=0.1 ** 9,
                    sparsity_cost=10 ** 8,
                    sparsity_decay=0.9,
                    epochs=5)

        config = rbm_config.RBMConfig()
        config.v_n = 3
        config.h_n = 2
        config.v_unit = rbm_units.GaussianVisibleUnit
        config.progress_logger = rbm_logger.ProgressLogger()
        config.train_params = tr
        np_rand = np.random.RandomState(123)

        # Weights
        W = np_rand.uniform(low=-1./10, high=-1./10, size=(3, 2)).astype(np.float32)
        vb = np.array([-0.1, 0, 0.1], dtype=np.float32)
        hb = np.array([0.01, -0.01], dtype=np.float32)
        Wt = theano.shared(W, name='W')
        vbt = theano.shared(vb, name='vbias')
        hbt = theano.shared(hb, name='hbias')
        g_rbm = rbm.RBM(config, W=Wt, h_bias=hbt, v_bias=vbt)
        self.assertTrue(g_rbm)
        self.assertTrue(isinstance(g_rbm.v_unit, rbm_units.GaussianVisibleUnit))
        self.assertTrue(isinstance(g_rbm.h_unit, rbm_units.RBMUnit))
        self.assertTrue(np.count_nonzero(g_rbm.W.get_value(borrow=True) - W) == 0)
        self.assertTrue(np.count_nonzero(g_rbm.v_bias.get_value(borrow=True) - vb) == 0)
        self.assertTrue(np.count_nonzero(g_rbm.h_bias.get_value(borrow=True) - hb) == 0)

        x = sklearn.preprocessing.scale(np.array([[200.0, 188., 7.],
                                                  [150.0, 128., 0.],
                                                  [250.0, 98., 3.]],
                                                 dtype=theano.config.floatX))
        v = theano.shared(x)
        _, _, h = g_rbm.sample_h_given_v(v)
        _, _, vs = g_rbm.sample_v_given_h(h)
        _, _, hs = g_rbm.sample_h_given_v(vs)
        dw = T.dot(v.T, h) - T.dot(vs.T, hs)
        dv = T.sum(v - vs, axis=0)
        dh = T.sum(h - hs, axis=0)
        gr = g_rbm.get_partial_derivatives(v, None)['gradients']
        gdw, gdv, gdh = gr[0], gr[1], gr[2]
        print gdw, gdv, gdh
        compute_derivative = theano.function([], [dw, dv, dh, gdw, gdv, gdh])
        for i in xrange(20):
            a, b, c, d, e, f = compute_derivative()
            # print a, b, c
            print d, e, f

    def test_digits(self):
        tr = rbm_config.TrainParam(learning_rate=0.05,
                    momentum_type=rbm_config.CLASSICAL,
                    momentum=0.5,
                    weight_decay=0,
                    sparsity_constraint=False,
                    sparsity_target=0.1 ** 9,
                    sparsity_cost=10 ** 8,
                    sparsity_decay=0.9,
                    epochs=5)

        config = rbm_config.RBMConfig()
        config.v_n = 784
        config.h_n = 100
        config.v_unit = rbm_units.GaussianVisibleUnit
        config.h_unit = rbm_units.ReLUnit
        config.progress_logger = rbm_logger.ProgressLogger()
        config.train_params = tr
        np_rand = np.random.RandomState(123)

        # Weights
        W = np_rand.uniform(low=-1./10, high=1./10, size=(784, 100)).astype(np.float32)
        vb = np.zeros(784, dtype=np.float32)
        hb = np.array(100, dtype=np.float32)
        Wt = theano.shared(W, name='W')
        vbt = theano.shared(vb, name='vbias')
        hbt = theano.shared(hb, name='hbias')
        g_rbm = rbm.RBM(config, W=Wt, h_bias=hbt, v_bias=vbt)
        self.assertTrue(g_rbm)
        self.assertTrue(isinstance(g_rbm.v_unit, rbm_units.GaussianVisibleUnit))
        self.assertTrue(isinstance(g_rbm.h_unit, rbm_units.RBMUnit))
        self.assertTrue(np.count_nonzero(g_rbm.W.get_value(borrow=True) - W) == 0)
        self.assertTrue(np.count_nonzero(g_rbm.v_bias.get_value(borrow=True) - vb) == 0)
        self.assertTrue(np.count_nonzero(g_rbm.h_bias.get_value(borrow=True) - hb) == 0)

        tr, vl, te = m_loader.load_digits(n=[5, 10, 10], pre={'scale': True})
        v = tr[0]

        # print 'inputs:'
        # table = ss.itemfreq(v.get_value(borrow=True))
        # x = [pt[0] for pt in table]
        # y = [pt[1] for pt in table]
        # plt.plot(x, y)
        # plt.show()

        # v = theano.shared(x)
        _, _, h = g_rbm.sample_h_given_v(v)
        _, _, vs = g_rbm.sample_v_given_h(h)
        _, _, hs = g_rbm.sample_h_given_v(vs)
        dw = T.dot(v.T, h) - T.dot(vs.T, hs)
        dv = T.sum(v - vs, axis=0)
        dh = T.sum(h - hs, axis=0)
        gr = g_rbm.get_partial_derivatives(v, None)['gradients']
        gdw, gdv, gdh = gr[0], gr[1], gr[2]
        print gdw, gdv, gdh
        compute_derivative = theano.function([], [dw, dv, dh, gdw, gdv, gdh])
        for i in xrange(1):
            a, b, c, d, e, f = compute_derivative()
            # print a, b, c
            print 'unfold'
            print a[0], b[1:5], c[1:5]

            print 'rbm'
            print d[0], e[1:5], f[1:5]

    def test_kanades(self):
        tr = rbm_config.TrainParam(learning_rate=0.05,
                    momentum_type=rbm_config.CLASSICAL,
                    momentum=0.5,
                    weight_decay=0,
                    sparsity_constraint=False,
                    sparsity_target=0.1 ** 9,
                    sparsity_cost=10 ** 8,
                    sparsity_decay=0.9,
                    epochs=5)

        nvis = 625
        nhid = 1000

        config = rbm_config.RBMConfig()
        config.v_n = nvis
        config.h_n = nhid
        config.v_unit = rbm_units.GaussianVisibleUnit
        config.h_unit = rbm_units.ReLUnit
        config.progress_logger = rbm_logger.ProgressLogger()
        config.train_params = tr
        np_rand = np.random.RandomState(123)

        # Weights
        W = np_rand.uniform(low=-1./10, high=1./10, size=(nvis, nhid)).astype(np.float32)
        vb = np.zeros(nvis, dtype=np.float32)
        hb = np.array(nhid, dtype=np.float32)
        Wt = theano.shared(W, name='W')
        vbt = theano.shared(vb, name='vbias')
        hbt = theano.shared(hb, name='hbias')
        g_rbm = rbm.RBM(config, W=Wt, h_bias=hbt, v_bias=vbt)
        self.assertTrue(g_rbm)
        self.assertTrue(isinstance(g_rbm.v_unit, rbm_units.GaussianVisibleUnit))
        self.assertTrue(isinstance(g_rbm.h_unit, rbm_units.RBMUnit))
        self.assertTrue(np.count_nonzero(g_rbm.W.get_value(borrow=True) - W) == 0)
        self.assertTrue(np.count_nonzero(g_rbm.v_bias.get_value(borrow=True) - vb) == 0)
        self.assertTrue(np.count_nonzero(g_rbm.h_bias.get_value(borrow=True) - hb) == 0)

        tr, vl, te = k_loader.load_kanade(n=100, pre={'scale': True})
        v = tr[0]

        print 'inputs:'
        table = ss.itemfreq(v.get_value(borrow=True))
        x = [pt[0] for pt in table]
        y = [pt[1] for pt in table]
        plt.plot(x, y)
        plt.show()

        # v = theano.shared(x)
        _, _, h = g_rbm.sample_h_given_v(v)
        _, _, vs = g_rbm.sample_v_given_h(h)
        _, _, hs = g_rbm.sample_h_given_v(vs)
        dw = T.dot(v.T, h) - T.dot(vs.T, hs)
        dv = T.sum(v - vs, axis=0)
        dh = T.sum(h - hs, axis=0)
        gr = g_rbm.get_partial_derivatives(v, None)['gradients']
        gdw, gdv, gdh = gr[0], gr[1], gr[2]
        print gdw, gdv, gdh
        compute_derivative = theano.function([], [dw, dv, dh, gdw, gdv, gdh])
        for i in xrange(1):
            a, b, c, d, e, f = compute_derivative()
            # print a, b, c
            print 'unfold'
            print a[0], b[1:5], c[1:5]

            print 'rbm'
            print d[0], e[1:5], f[1:5]


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

    def test_kanade_scaling(self):
        datasets = k_loader.load_kanade(shared=False, n=10)
        x, y = datasets[0]
        xsc = sklearn.preprocessing.scale(x.astype(float))

        print x
        print xsc

        datasets = k_loader.load_kanade(shared=False, n=10, pre={'scale2unit':True, 'scale':True})
        x, y = datasets[0]
        xsc = sklearn.preprocessing.scale(x.astype(float))
        print xsc

        # print np.mean(xsc, axis=0)
        # print np.std(xsc, axis=0)

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
        train, valid, test = m_loader.load_digits(n=[100, 0, 100], pre={'scale':True})
        train_x, train_y = train
        valid_x, valid_y = valid
        test_x, test_y = test

        tr = rbm_config.TrainParam(learning_rate=0.0001,
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
                                       progress_logger=rbm_logger.ProgressLogger())

        curr_dir = store.move_to('simple_gaussian_rbm_test')
        print "... moved to {}".format(curr_dir)

        # Train RBM
        gaussian_rbm.train(train_x)

        # Test RBM
        # gaussian_rbm.plot_samples(test_x)


if __name__ == '__main__':
    print "Test Utilities"
    unittest.main()