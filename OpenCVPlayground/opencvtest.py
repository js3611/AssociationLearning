__author__ = 'joschlemper'

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Read images

print os.getcwd()

img = cv2.imread('OpenCVPlayground/S010_004_00000019.png', 0)
print img

if np.any(img):
    cv2.imshow('image', img)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',img)
        cv2.destroyAllWindows()

    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()