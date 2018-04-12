#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:44:24 2018

@author: Akash
"""

import numpy as np
import cv2 # opencv
import os # control and access the directory structure in local machine
from matplotlib import pyplot as plt
import time
from sklearn.cross_validation import cross_val_score,cross_val_predict
from sklearn import metrics

from sklearn import cross_validation as cval
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_score
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

os.chdir('/Users/akash/Downloads/Face_final') #folder where I unzipped data.zip
haarcascades_path = os.listdir('/Users/akash/Desktop')
frontface_alt_cascade = '/Users/akash/Desktop/haarcascades/haarcascade_frontalface_alt.xml'
frontface_default_cascade = '/Users/akash/Desktop/haarcascades/haarcascade_frontalface_default.xml'
frontface_alt2_cascade = '/Users/akash/Desktop/haarcascades/haarcascade_frontalface_alt2.xml'
frontface_alt_tree_cascade = '/Users/akash/Desktop/haarcascades/haarcascade_frontalface_alt2.xml'

def detect(faceCascade, gray_,  scaleFactor_ = 1.1):
    faces = faceCascade.detectMultiScale(
                    gray_,
                    scaleFactor= scaleFactor_,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
    return faces

# code that iterates thru the images in the celebrityfaces dataset and detects faces. Finally it
# ... only displays those images that it can't detect the faces.
faceCascade_default = cv2.CascadeClassifier(frontface_default_cascade)
faceCascade_alt = cv2.CascadeClassifier(frontface_alt_cascade)
faceCascade_alt2 = cv2.CascadeClassifier(frontface_alt2_cascade)
faceCascade_alt_tree = cv2.CascadeClassifier(frontface_alt_tree_cascade)

face_train = os.listdir('New_folder/train/')
y_train = []
for imgfolder in os.listdir('New_folder/train/'):
    if(imgfolder != '.DS_Store'):
        for filename in os.listdir('New_folder/train/' + imgfolder):# iterate thru each image in a celeb folder
            filename = 'New_folder/train/' + imgfolder + '/' + filename # build the path to the image file
            if(filename.endswith('.jpg')):
                y_train.append(imgfolder)
y_train = np.asarray(y_train)


X_images = []
for imgfolder in os.listdir('New_folder/train/'):
    if(imgfolder != '.DS_Store'):
        for filename in os.listdir('New_folder/train/' + imgfolder):
            filename = 'New_folder/train/' + imgfolder + '/' + filename
            if(filename.endswith('.jpg')):
        #print(filename)
                img = cv2.imread(filename,0)
                img = cv2.resize(img, (115,170), interpolation = cv2.INTER_AREA)
X_images.append(img)
X_images = np.asarray(X_images)

X_data = X_images.reshape(X_images.shape[0], X_images.shape[1] * X_images.shape[2])

c = 0
X,y = [], []
for dirname, dirnames, filenames in os.walk(path):
    for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                    c = c+1
                    return [X,y]

scores = cross_val_score(model, df, y, cv=6)












