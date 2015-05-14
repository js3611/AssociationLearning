__author__ = 'joschlemper'

import unittest
import cv2
import numpy as np
import kanade_loader as loader
from matplotlib import pyplot as plt

class MyTestCase(unittest.TestCase):
    def test_sth(self):
        self.assertTrue(True)

    def test_load(self):
        dataset = loader.load_kanade(shared=False, resolution='50_50')
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
        dataset = loader.load_kanade(shared=False, resolution='50_50', emotions=['anger'])
        self.assertTrue(np.unique(dataset[1]) == 1)

        dataset = loader.load_kanade(shared=False, resolution='50_50', emotions=['anger', 'contempt'])
        labels = np.unique(dataset[1])
        self.assertTrue(len(labels) == 2 and 1 in labels and 2 in labels)

    def test_load_n(self):
        dataset = loader.load_kanade(shared=False, resolution='50_50', n=100)
        self.assertTrue(len(dataset[0]) == 100)

if __name__ == '__main__':
    unittest.main()
