import unittest
from rbm import RBM
from rbm_config import *
from rbm_logger import *
from utils import *
import theano.tensor as T
import numpy as np
import scipy.stats as ss
theano.config.optimizer = 'None'


class AssociativeRBMTest(unittest.TestCase):
    def setUpAssociativeRBM(self, v=5, v2=5, h=10):
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
        config.progress_logger = AssociationProgressLogger()
        self.rbm = RBM(config)
        self.x = np.array([[1, 1, 1, 0, 0]], dtype=t_float_x)
        self.y = np.array([[0, 0, 0, 0, 1]], dtype=t_float_x)
        self.x2 = np.array([[1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1]],
                           dtype=t_float_x)
        self.y2 = np.array([[0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1]],
                           dtype=t_float_x)
        self.yl = np.array([[0, 0],
                            [0, 1]],
                           dtype=t_float_x)

        self.tx = theano.shared(np.array([[1, 1, 1, 0, 0]], dtype=t_float_x))
        self.ty = theano.shared(np.array([[0, 0, 0, 0, 1]], dtype=t_float_x))
        self.tz = theano.shared(np.array([[0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 1]], dtype=t_float_x))

        self.tl = theano.shared(np.array([[0, 1],
                                          [0, 1],
                                          [0, 1],
                                          [0, 1],
                                          [1, 0]], dtype=t_float_x))


    def test_parameters_order(self):
        self.setUpAssociativeRBM()
        rbm = self.rbm
        self.assertEqual(str(rbm.params[0]), 'W')
        self.assertEqual(str(rbm.params[1]), 'v_bias')
        self.assertEqual(str(rbm.params[2]), 'h_bias')
        self.assertEqual(str(rbm.params[3]), 'U')
        self.assertEqual(str(rbm.params[4]), 'v_bias2')

    def test_free_energy(self):
        self.setUpAssociativeRBM()
        rbm = self.rbm
        w = rbm.W.get_value(borrow=True)
        u = rbm.U.get_value(borrow=True)
        v = T.dmatrix("v")
        v2 = T.dmatrix("v2")
        v_bias = rbm.v_bias.eval()
        v_bias2 = rbm.v_bias2.eval()
        h_bias = rbm.h_bias.eval()

        res = rbm.free_energy(v, v2)
        f = theano.function([v, v2], [res])
        theano_res = f(self.x, self.y)

        # Test for case only v1 is present
        n1 = - np.dot(self.x, v_bias)
        n2 = - np.dot(self.y, v_bias2)
        n3 = - np.sum(np.log(1 + np.exp(h_bias + np.dot(self.x, w) + np.dot(self.y, u))))
        np_res = n1 + n2 + n3

        print theano_res
        print np_res

        diff = theano_res == np_res
        self.assertTrue(np.all(diff))

    def test_negative_statistics(self):
        self.setUpAssociativeRBM(5, 2, 10)
        rbm = self.rbm
        x = T.matrix("x")
        y = T.matrix("y")
        res = rbm.negative_statistics(x, y)
        updates = res[0]

        # Returns chain end
        f = theano.function([x, y], res[1:], updates=updates)

        # sample, v, vp, vs, h, hp, hs = f(self.x)
        # print sample
        # print vs

        result = f(self.y2, self.yl)
        # print result[0]

        pass

    def test_partial_derivatives(self):
        self.setUpAssociativeRBM()
        rbm = self.rbm
        x = T.dmatrix("x")
        y = T.dmatrix("y")

        grad_meta = rbm.get_partial_derivatives(x, y)
        gradients = grad_meta["gradients"]
        updates = grad_meta["updates"]
        v_total_inputs = grad_meta["statistics"]
        g_W, g_v, g_h, g_U, g_v2= gradients
        f = theano.function([x, y], [g_W, g_v, g_h, g_U, g_v2], updates=updates)
        g_W, g_v, g_h, g_U, g_v = f(self.x, self.y)

        print g_W
        print g_v
        print g_h
        print g_U
        print g_v2
        pass

    def test_get_train_fn(self):
        self.setUpAssociativeRBM(5, 2, 10)
        rbm = self.rbm
        fn = rbm.get_train_fn(self.tz, self.tl)
        res = fn(0)
        print res

        pass

