__author__ = 'joschlemper'

import unittest
import cv2
import numpy as np
import theano
import kanade_loader as loader
from matplotlib import pyplot as plt
from scipy.stats import itemfreq

class MyTestCase(unittest.TestCase):
    def test_sth(self):
        self.assertTrue(True)

    def test_load(self):
        dataset = loader.load_kanade(shared=False, set_name='50_50')
        x = dataset[0]
        y = dataset[1]

        print x.shape

        self.assertTrue(len(x) == len(y))
        self.assertTrue(len(x[0]) == 50*50)

        # img = x[0].reshape(50,50)
        # plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

    def test_filter(self):
        dataset = loader.load_kanade(shared=False, set_name='50_50', emotions=['anger'])
        self.assertTrue(np.unique(dataset[1]) == 1)

        dataset = loader.load_kanade(shared=False, set_name='50_50', emotions=['anger', 'contempt'])
        labels = np.unique(dataset[1])
        self.assertTrue(len(labels) == 2 and 1 in labels and 2 in labels)

    def test_load_n(self):
        dataset = loader.load_kanade(shared=False, set_name='50_50', n=100)
        self.assertTrue(len(dataset[0]) == 100)

    def test_scale(self):
        dataset = loader.load_kanade(shared=False, n=2, pre={'scale2unit':True})

        self.assertTrue(len(dataset[0]) == 2 and len(dataset[1]) == 2)

        print dataset[0]
        print itemfreq(dataset[0])

    def test_shared(self):
        dataset = loader.load_kanade(n=10)
        train_x, train_y = dataset

        # print type(train_x)
        # print type(train_y)
        self.assertTrue(isinstance(train_x, theano.tensor.sharedvar.TensorSharedVariable))

if __name__ == '__main__':
    unittest.main()
