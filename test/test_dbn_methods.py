__author__ = 'joschlemper'

import unittest
import rbm
import dbn
import theano
import theano.tensor as T
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_bottom_up(self):
        x = np.array([[1,2,3,4,5],[1,2,3,4,5]])
        x2 = np.array([[1,2,3,4]])

        dbn = dbn.DBN(topology=[5, 4, 3])

        # x_t = theano.shared(x)

        res = dbn.bottom_up_pass(x)
        res2 = dbn.bottom_up_pass(x, 0, 1)
        res3 = dbn.bottom_up_pass(x2, 1, 2)

        print res
        print res2
        print res3

        self.assertTrue(res.shape[1] == 3)
        self.assertTrue(res2.shape[1] == 4)
        self.assertTrue(res3.shape[1] == 3)

        r1 = dbn.top_down_pass(res)
        r2 = dbn.top_down_pass(res, 2, 1)
        r3 = dbn.top_down_pass(res2, 1, 0)

        self.assertTrue(r1.shape[1] == 5)
        self.assertTrue(r2.shape[1] == 4)
        self.assertTrue(r3.shape[1] == 5)

if __name__ == '__main__':
    unittest.main()
