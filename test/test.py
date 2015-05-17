import unittest
from rbm import RBM
from rbm_config import *
from rbm_logger import *
from activationFunction import *
from theano.tensor.shared_randomstreams import RandomStreams
from utils import *
import numpy as np
import theano
import theano.tensor as T
import numpy as np
import scipy.stats as ss
theano.config.optimizer = 'None'

class RBMMethodTest(unittest.TestCase):
    def setUpSimpleRBM(self):
        v = 2
        v2 = 2
        h = 10
        tr = TrainParam(learning_rate=0.01,
                        momentum_type=CLASSICAL,
                        momentum=0.5,
                        weight_decay=0.01,
                        sparsity_constraint=False,
                        batch_size=1,
                        epochs=5)

        config = RBMConfig()
        config.associative = True
        config.v_n = v
        config.v2_n = v2
        config.h_n = h
        config.train_params = tr
        config.progress_logger = ProgressLogger()
        self.rbm = RBM(config)

    def setUpRBM(self):
        v = 5
        v2 = 5
        h = 10
        tr = TrainParam(learning_rate=0.01,
                        momentum_type=CLASSICAL,
                        momentum=0.5,
                        weight_decay=0.01,
                        sparsity_constraint=False,
                        batch_size=1,
                        epochs=5)

        config = RBMConfig()
        config.associative = True
        config.v_n = v
        config.v2_n = v2
        config.h_n = h
        config.train_params = tr
        config.progress_logger = ProgressLogger()
        self.rbm = RBM(config)
        self.rbmx1 = np.array([[2, 5, 5, 2, 1]], dtype=t_float_x)
        self.rbmx2 = np.array([[5, 2, 3, 8, 1]], dtype=t_float_x)

    def setUp(self):
        # datasets = load_data('mnist.pkl.gz')
        # train_set_x, train_set_y = datasets[0]
        # valid_set_x, valid_set_y = datasets[1]
        # test_set_x, test_set_y = datasets[2]
        #
        # self.tr_x = train_set_x
        # self.tr_y = train_set_y
        # self.vl_x = valid_set_x
        # self.vl_y = valid_set_y
        # self.te_x = test_set_x
        # self.te_y = test_set_y

        # Set up:
        # Dim(W)  = (2, 3) = (v, h)
        # Dim(c)  = (1, 3) = (1, h)
        # Dim(x1) = (1, 2) = (n, v)
        # Dim(x2) = (5, 2) = (n, v)

        self.W = np.array([[1, 2, 3], [4, 5, 6]], dtype=t_float_x)
        self.U = np.array([[1, 2, 3], [4, 5, 6]], dtype=t_float_x)
        self.c = np.array([1, 10, 100], dtype=t_float_x)
        self.b1 = np.array([-5, 5], dtype=t_float_x)
        self.b2 = np.array([-10, 3], dtype=t_float_x)
        self.x1 = np.array([[2, 5]], dtype=t_float_x)
        self.x12 = np.array([[5, 2]], dtype=t_float_x)
        self.x2 = np.array([[3, 2], [3, 4], [6, 5], [2, 2], [1, 1]], dtype=t_float_x)
        self.x3 = np.array([[3, 9], [3, 3], [1, 5], [7, 2], [1, 1]], dtype=t_float_x)

    def test_simple(self):
        W = T.dmatrix('W')
        x = T.dmatrix('x')
        c = T.dvector('c')
        r = T.dot(x, W) + c
        f = theano.function([x, W, c], [r])

        # 1
        theano_res = f(self.x1, self.W, self.c)
        np_res = np.dot(self.x1, self.W) + self.c
        diff = theano_res == np_res
        self.assertTrue(np.all(diff))

        # 2
        theano_res = f(self.x2, self.W, self.c)
        np_res = np.dot(self.x2, self.W) + self.c
        diff = theano_res == np_res
        self.assertTrue(np.all(diff))

    def test_given(self):
        W = T.dmatrix('W')
        x = T.dmatrix('x')
        c = T.dvector('c')
        r = T.dot(x, W) + c
        f = theano.function([], [r], givens={W: self.W, x: self.x1, c: self.c})

        # Test 1
        theano_res = f()
        np_res = np.dot(self.x1, self.W) + self.c
        diff = theano_res == np_res
        self.assertTrue(np.all(diff))

    def test_combination(self):
        W = self.W
        x = T.dmatrix('x')
        b = T.dvector('b')
        c = T.dot(x, W) + b
        f = theano.function([], [c], givens={x: self.x1, b: self.c})

        # Test 1
        theano_res = f()
        np_res = np.dot(self.x1, self.W) + self.c
        diff = theano_res == np_res
        self.assertTrue(np.all(diff))

    def test_free_energy(self):
        w = self.W
        u = self.U
        v = T.dmatrix("v")
        v2 = T.dmatrix("v2")
        v_bias = self.b1
        v_bias2 = self.b2
        h_bias = self.c

        t1 = - T.dot(v, v_bias)
        t2 = - T.dot(v2, v_bias2)
        t3 = - T.sum(T.log(1 + T.exp(h_bias + T.dot(v2, u) + T.dot(v, w))))
        free_energy = t1 + t2 + t3
        f = theano.function([v, v2], [free_energy])
        theano_res = f(self.x2, self.x3)

        n1 = - np.dot(self.x2, v_bias)
        n2 = - np.dot(self.x3, v_bias2)
        n3 = - np.sum(np.log(1 + np.exp(h_bias + np.dot(self.x3, u) + np.dot(self.x2, w))))
        np_res = n1 + n2 + n3

        diff = theano_res == np_res
        self.assertTrue(np.all(diff))

    def test_prop_up(self):
        self.setUpSimpleRBM()

        rbm = self.rbm
        v1 = T.dmatrix("v1")
        v2 = T.dmatrix("v2")
        # Test Single
        out = rbm.prop_up(v1)
        out_fn = theano.function([], [out[0], out[1]], givens={v1: self.x1})
        out_sum, out_sum_mapped = out_fn()

        h_sum = np.dot(self.x1, rbm.W.get_value(borrow=True)) + rbm.h_bias.eval()
        h_sum_mapped = 1 / (1 + np.exp(-h_sum))
        self.assertTrue(np.all(out_sum == h_sum))
        self.assertTrue((np.all(out_sum_mapped == h_sum_mapped)))


        # Test Double
        out = rbm.prop_up(v1, v2)
        out_fn = theano.function([], [out[0], out[1]], givens={v1: self.x1, v2: self.x12})
        out_sum, out_sum_mapped = out_fn()
        h_sum = np.dot(self.x1, rbm.W.get_value(borrow=True)) + np.dot(self.x12, rbm.U.get_value(
            borrow=True)) + rbm.h_bias.eval()
        h_sum_mapped = 1 / (1 + np.exp(-h_sum))
        # h_sum_mapped = theano.function([], [log_sig(h_sum)])()

        self.assertTrue(np.all(out_sum == h_sum))
        self.assertTrue((np.all(out_sum_mapped == h_sum_mapped)))

    def test_prop_down(self):
        self.setUpRBM()
        self.assertTrue(self.rbm.h_n == 10)
        rbm = self.rbm
        W = rbm.W.get_value(borrow=True)
        U = rbm.U.get_value(borrow=True)
        v1 = T.dmatrix("v1")
        v2 = T.dmatrix("v2")
        h = np.array([[1, 2, 3, 4, 5, -1, -2, -3, -4, -5]])

        # Single
        x = T.dmatrix("x")
        out = rbm.prop_down(x)
        f = theano.function([x], out)
        out_sum, out_sum_mapped = f(h)
        h_sum = np.dot(h, W.T) + rbm.v_bias.eval()
        h_sum_mapped = theano.function([], [log_sig(h_sum)])()
        self.assertTrue(np.all(out_sum == h_sum))
        self.assertTrue(np.all(out_sum_mapped == h_sum_mapped))

        # Assoc
        out = rbm.prop_down_assoc(x)
        f = theano.function([x], out)
        out_sum, out_sum_mapped = f(h)
        h_sum2 = np.dot(h, U.T) + rbm.v_bias2.eval()
        h_sum_mapped2 = theano.function([], [log_sig(h_sum2)])()
        self.assertTrue(np.all(out_sum == h_sum2))
        self.assertTrue(np.all(out_sum_mapped == h_sum_mapped2))

    def test_deterministic_reconstruct_scan(self):
        self.setUpRBM()
        self.assertTrue(self.rbm.h_n == 10)
        rbm = self.rbm
        W = rbm.W.get_value(borrow=True)
        U = rbm.U.get_value(borrow=True)
        vb1 = rbm.v_bias.eval()
        vb2 = rbm.v_bias2.eval()
        hb = rbm.h_bias.eval()

        # Initial values
        rand = np.random.RandomState(123)
        rand = RandomStreams(rand.randint(2 ** 30))
        x1 = self.rbmx1
        x2 = rand.binomial(size=self.rbmx2.shape, n=1, p=0.5, dtype=t_float_x).eval()

        def gibbs(ux, u2):
            h, hp = rbm.prop_up(ux, u2)
            v, vp = rbm.prop_down(hp)
            v2, v2p = rbm.prop_down_assoc(hp)
            return [h, hp, v, ux, v2, v2p]


        # THEANO
        x = T.dmatrix("x")
        y = T.dmatrix("y")
        x_start = x
        y_start = y
        (
            res,
            updates
        ) = theano.scan(
            gibbs,
            outputs_info=[None, None, None,
                          x_start, None, y_start],
            n_steps=5
        )
        f = theano.function([x, y], res, updates=updates)

        rand = np.random.RandomState(1234)
        rand = RandomStreams(rand.randint(2 ** 30))
        rbm.rand = rand
        [h, hp, v, vp, v2, v2p] = f(self.rbmx1, x2)
        # print h
        # print hp
        # print "h: \n{}".format(h)
        # print "hp: \n{}".format(hp)
        # # print "hs: \n{}".format(hs)
        # print "v: \n{}".format(v)
        # print "vp: \n{}".format(vp)
        # # print "vs: \n{}".format(vs)
        # print "v2: \n{}".format(v)
        # print "v2p: \n{}".format(v2p)
        # # print "v2s: \n{}".format(v2s)
        # print "Numpy"
        # =============== NUMPY ================

        rand = np.random.RandomState(1234)
        rand = RandomStreams(rand.randint(2 ** 30))
        for i in xrange(0, 5):
            # Sample h
            h, ph = np_prop_up(x1, W, hb, x2, U)
            # sample using same seed
            #hs = rand.binomial(size=ph.shape, n=1, p=ph, dtype=t_float_x).eval()
            # print h

            # Sample x, x2
            u, pu = np_prop_down(ph, W, vb1)
            # dummy call, just to adjust seed
            #us = rand.binomial(size=pu.shape, n=1, p=pu, dtype=t_float_x).eval()

            u2, pu2 = np_prop_down(ph, U, vb2)
            x2 = pu2
            #x2 = rand.binomial(size=pu2.shape, n=1, p=pu2, dtype=t_float_x).eval()
            # print x2
            # print x2

        self.assertTrue(np.all(v2p[-1] == x2))

    def test_indeterministic_reconstruct_scan(self):
        self.setUpRBM()
        self.assertTrue(self.rbm.h_n == 10)
        rbm = self.rbm
        W = rbm.W.get_value(borrow=True)
        U = rbm.U.get_value(borrow=True)
        vb1 = rbm.v_bias.eval()
        vb2 = rbm.v_bias2.eval()
        hb = rbm.h_bias.eval()
        k = 100

        # Initial values
        rand = np.random.RandomState(123)
        rand = RandomStreams(rand.randint(2 ** 30))
        x1 = self.rbmx1
        x2 = rand.binomial(size=self.rbmx2.shape, n=1, p=0.5, dtype=t_float_x).eval()

        def gibbs(ux, u2):
            h, hp = rbm.prop_up(ux, u2)
            hs = rbm.rand.binomial(size=hp.shape, n=1, p=hp, dtype=t_float_x)
            v, vp = rbm.prop_down(hs)
            vs = rbm.rand.binomial(size=vp.shape, n=1, p=vp, dtype=t_float_x)
            v2, v2p = rbm.prop_down_assoc(hs)
            v2s = rbm.rand.binomial(size=v2p.shape, n=1, p=v2p, dtype=t_float_x)
            return [h, hp, hs, v, v2p, ux, v2, v2p, v2s]


        # THEANO
        x = T.dmatrix("x")
        y = T.dmatrix("y")
        x_start = x
        y_start = y
        (
            res,
            updates
        ) = theano.scan(
            gibbs,
            outputs_info=[None, None, None, None, None,
                          x_start, None, None, y_start],
            n_steps=k
        )
        f = theano.function([x, y], res, updates=updates)

        rand = np.random.RandomState(1234)
        rand = RandomStreams(rand.randint(2 ** 30))
        rbm.rand = rand
        [h, hp, hs, v, vp, vs, v2, v2p, v2s] = f(self.rbmx1, x2)
        # print h
        # print hp
        # print "h: \n{}".format(h)
        # print "hp: \n{}".format(hp)
        # print "hs: \n{}".format(hs)
        # print "v: \n{}".format(v)
        # print "vp: \n{}".format(vp)
        # print "vs: \n{}".format(vs)
        # print "v2: \n{}".format(v)
        # print "v2p: \n{}".format(v2p)
        # print "v2s: \n{}".format(v2s)

        # =============== NUMPY ================

        rand = np.random.RandomState(1234)
        rand = RandomStreams(rand.randint(2 ** 30))
        for i in xrange(0, k):
            # Sample h
            h, ph = np_prop_up(x1, W, hb, x2, U)
            # sample using same seed
            hs = rand.binomial(size=ph.shape, n=1, p=ph, dtype=t_float_x).eval()
            # print h

            # Sample x, x2
            u, pu = np_prop_down(hs, W, vb1)
            # dummy call, just to adjust seed
            us = rand.binomial(size=pu.shape, n=1, p=pu, dtype=t_float_x).eval()

            u2, pu2 = np_prop_down(hs, U, vb2)
            x2 = pu2
            x2 = rand.binomial(size=pu2.shape, n=1, p=pu2, dtype=t_float_x).eval()
            # print x2

        # self.assertTrue(np.all(v2p[-1] == x2))

    def test_indeterministic_reconstruct_scan_vs_theano(self):
        self.setUpRBM()
        self.assertTrue(self.rbm.h_n == 10)
        rbm = self.rbm
        W = rbm.W.get_value(borrow=True)
        U = rbm.U.get_value(borrow=True)
        vb1 = rbm.v_bias.eval()
        vb2 = rbm.v_bias2.eval()
        hb = rbm.h_bias.eval()
        k = 100

        # Initial values
        rand = np.random.RandomState(123)
        rand = RandomStreams(rand.randint(2 ** 30))
        x1 = self.rbmx1
        x2 = rand.binomial(size=self.rbmx2.shape, n=1, p=0.5, dtype=t_float_x).eval()

        def gibbs(ux, u2):
            h, hp = rbm.prop_up(ux, u2)
            hs = rbm.rand.binomial(size=hp.shape, n=1, p=hp, dtype=t_float_x)
            v, vp = rbm.prop_down(hs)
            vs = rbm.rand.binomial(size=vp.shape, n=1, p=vp, dtype=t_float_x)
            v2, v2p = rbm.prop_down_assoc(hs)
            v2s = rbm.rand.binomial(size=v2p.shape, n=1, p=v2p, dtype=t_float_x)
            return [h, hp, hs, v, v2p, ux, v2, v2p, v2s]


        # THEANO
        x = T.dmatrix("x")
        y = T.dmatrix("y")
        x_start = x
        y_start = y
        (
            res,
            updates
        ) = theano.scan(
            gibbs,
            outputs_info=[None, None, None, None, None,
                          x_start, None, None, y_start],
            n_steps=k
        )
        f = theano.function([x, y], res, updates=updates)

        rand = np.random.RandomState(1234)
        rand = RandomStreams(rand.randint(2 ** 30))
        rbm.rand = rand
        [h, hp, hs, v, vp, vs, v2, v2p, v2s] = f(self.rbmx1, x2)
        # print h
        # print hp
        # print "h: \n{}".format(h)
        # print "hp: \n{}".format(hp)
        # print "hs: \n{}".format(hs)
        # print "v: \n{}".format(v)
        # print "vp: \n{}".format(vp)
        # print "vs: \n{}".format(vs)
        # print "v2: \n{}".format(v)
        # print "v2p: \n{}".format(v2p)
        # print "v2s: \n{}".format(v2s)

        # =============== NUMPY ================

        rand = np.random.RandomState(1234)
        rand = RandomStreams(rand.randint(2 ** 30))
        for i in xrange(0, k):
            # Sample h
            h, ph = np_prop_up(x1, W, hb, x2, U)
            # sample using same seed
            hs = rand.binomial(size=ph.shape, n=1, p=ph, dtype=t_float_x).eval()
            # print h

            # Sample x, x2
            u, pu = np_prop_down(hs, W, vb1)
            # dummy call, just to adjust seed
            us = rand.binomial(size=pu.shape, n=1, p=pu, dtype=t_float_x).eval()

            u2, pu2 = np_prop_down(hs, U, vb2)
            x2 = pu2
            x2 = rand.binomial(size=pu2.shape, n=1, p=pu2, dtype=t_float_x).eval()
            # print x2

        # self.assertTrue(np.all(v2p[-1] == x2))

    def test_negative_statistics(self):
        pass


class NumpyTest(unittest.TestCase):
    def test_logical_index(self):
        a = np.array([0, 1,2,3,4,5,0, 1,2,3,4,5])
        idx = (a == 1) | (a == 0)
        # print a[idx]


def np_prop_up(v, w, b, v2=None, u=None):
    h_sum = np.dot(v, w) + b
    if np.any(v2) and np.any(u):
        h_sum += np.dot(v2, u)
    h_sum_mapped = 1 / (1 + np.exp(-h_sum))
    return h_sum, h_sum_mapped


def np_prop_down(h, w, b):
    h_sum = np.dot(h, w.T) + b
    h_sum_mapped = 1 / (1 + np.exp(-h_sum))
    return h_sum, h_sum_mapped


if __name__ == '__main__':
    unittest.main()
