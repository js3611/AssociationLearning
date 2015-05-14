__author__ = 'joschlemper'

import unittest
import cv2

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_load(self):
        img = cv2.imread('test_image/S010_004_00000019.png', 0)
        if img.any():
            print 'loaded'


if __name__ == '__main__':
    unittest.main()
