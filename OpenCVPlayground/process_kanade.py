__author__ = 'joschlemper'

import cv2
import os
import numpy as np


''' This file processes CK+ database and create a dataset out of it. '''
''' Pre-processing can be performed before image is cropped '''

def process_image(img):
    # img -> dst
    dst = img
    # dst = cv2.resize(img, (50, 50))
    return dst


# ----- Main Script ----- #
emo_dir_path = 'Emotion'
img_dir_path = 'extended-cohn-kanade-images'
img_shape = (50, 50)
resolution = '50_50'
emotion_dict = {'anger': 1, 'contempt': 2, 'disgust': 3, 'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7}
emotion_rev_dict = {1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Prepare dst directories
ROOT_DIR = os.getcwd()
if not os.path.isdir(resolution):
    os.mkdir(resolution)
paths = {}
for emotion in emotion_dict.keys():
    path = os.path.join(resolution, emotion)
    if not os.path.isdir(path):
        os.mkdir(path)
    paths[emotion] = path



# Go through Emotion to first get  list(img_name, emotion)
assert os.path.isdir(emo_dir_path)
os.chdir(emo_dir_path)
tuple_list = []
for subject in os.listdir('.'):
    for emo in os.listdir(subject):
        for file_name in os.listdir(os.path.join(subject, emo)):
            if file_name.endswith('.txt'):
                file_path = os.path.join(subject, emo, file_name)
                with open(file_path, 'r') as f:
                    label = int(f.readline().strip().split('.')[0])

                img_name = file_path.replace('_emotion.txt', '.png')
                tuple_list.append((img_name, label))
os.chdir('..')

print '{} images are will now be converted'.format(len(tuple_list))

assert os.path.isdir(img_dir_path)
# os.chdir(img_dir_path)

counter = 0
for img_name, label in tuple_list:
    # set dst path
    dst = paths[emotion_rev_dict[label]]
    
    # open image in grey scale
    img = cv2.imread(os.path.join(img_dir_path, img_name), 0)

    # detect face
    faces = face_detector.detectMultiScale(img)

    for face in faces:
        (x, y, w, h) = face
        # process image
        transformed = process_image(img[y:y+h, x:x+h])
        # save the image

        name = os.path.join(dst,str(counter)) + '.png'
        print name
        cv2.imwrite(name, transformed)
        counter += 1



