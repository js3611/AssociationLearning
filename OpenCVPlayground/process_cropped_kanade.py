__author__ = 'joschlemper'

import os
import cv2
import numpy as np


# ----- Main Script ----- #
noise_level = 0.9
dst = 'noise{}_'.format(noise_level)
resize = True
n = 25

img_shape = (n, n)
resolution = '_'.join(map(str, [n, n]))
src = 'orig'
dst += resolution if resize else ''
emotion_dict = {'anger': 1, 'contempt': 2, 'disgust': 3, 'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7}
emotion_rev_dict = {1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}

# Go through Emotion to first get  list(img_name, emotion)
assert os.path.isdir(src)
if not os.path.isdir(dst):
    os.mkdir(dst)
    for emotion in emotion_dict.keys():
        os.mkdir(os.path.join(dst, emotion))

counter = 0


def process_image_sp_and_rotation(original):    
    rows, cols = original.shape
    # rotate
    img_set = []
    # for rotation_degree in np.arange(-5, 5, 1):
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_degree, 1)
    #     dst_img = cv2.warpAffine(original, M, (cols, rows))
    #     img_set.append(dst_img)
    img_set.append(original)
    
    # salt & pepper noise - add 3 copies of the orig image with 0.01%, 0.05%, 0.1% noise
    noisy_img_set = []
    for img in img_set:
        # for noise_lvl in [0.001, 0.003, 0.005]:
        for noise_lvl in [noise_level]:
            salt_mask = np.random.binomial(1, noise_lvl / 2, (cols, rows)).astype(np.int8)
            pepper_mask = np.random.binomial(1, noise_lvl / 2, (cols, rows)).astype(np.int8)
            sp_mask = abs(np.maximum(salt_mask, pepper_mask) - 1)
            noisy_img = 255 * salt_mask + sp_mask * img
            noisy_img_set.append(noisy_img)

    # save them
    # return img_set + noisy_img_set
    return noisy_img_set

def process_image_equilize(original):
    edged = cv2.equalizeHist(original)
    return [edged]


def process_image_edge_detect(original):
    edged = cv2.Canny(original, 150, 150)
    return [edged]


def process_image_sharpen_edge(original):
    blur = cv2.GaussianBlur(original, (5, 5), 0)
    return [cv2.addWeighted(original, 1.5, blur, -0.5, 0)]
    # return [original]

for emotion in emotion_dict.keys():
    img_path = os.path.join(src, emotion)
    for img_name in os.listdir(img_path):
        if not img_name.endswith('png'):
            continue

        print 'transforming {}'.format(img_name)

        # Read image
        img_file = os.path.join(img_path, img_name)
        orig_img = cv2.imread(img_file, 0)

        # images = process_image_edge_detect(orig_img)
        images = process_image_equilize(orig_img)
        images = process_image_sharpen_edge(images[0])
        images = process_image_sp_and_rotation(images[0])
        
        # print len(images)
        for img in images:
            if resize:
                img = cv2.resize(img, img_shape)
            name = os.path.join(dst, emotion, str(counter)) + '.png'
            cv2.imwrite(name, img)
            # print 'saved {}'.format(name)
            counter += 1

