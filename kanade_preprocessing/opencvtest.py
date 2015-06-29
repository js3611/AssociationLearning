__author__ = 'joschlemper'

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Read images
print os.getcwd()

# 0 is for black and white
img = cv2.imread('jo.jpg', 0)

# Detect face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Transform image
crop_img = []
faces = face_cascade.detectMultiScale(img)
# faces = face_cascade.detectMultiScale(img, 1.3, 5)
for (x, y, w, h) in faces:
    assert w == h
    print 'face detected'
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    crop_img = img[y:y+h, x:x+h]

reduced_crop_img = cv2.resize(crop_img, (50, 50))
cv2.imshow('image', reduced_crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('cropped_jo_50_50.png', reduced_crop_img)

# if np.any(img):
#
#     k = cv2.waitKey(0)
#     if k == 27:         # wait for ESC key to exit
#         cv2.destroyAllWindows()
#     elif k == ord('s'): # wait for 's' key to save and exit
#         cv2.imwrite('messigray.png',img)
#         cv2.destroyAllWindows()
#
#     plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     plt.show()