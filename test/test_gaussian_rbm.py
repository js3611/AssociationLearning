import unittest
import datastorage as store
import utils
import rbm as rbm
import DBN as DBN


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.gaussian_rbm = rbm.GaussianRBM()

    def test_set_up(self):
        self.assertTrue(self.gaussian_rbm)

    def test_subclass(self):
        self.assertTrue(self.gaussian_rbm.mu == 0)
        self.assertTrue('gaussian' in str(self.gaussian_rbm))


if __name__ == '__main__':
    print "Test Utilities"
    unittest.main()