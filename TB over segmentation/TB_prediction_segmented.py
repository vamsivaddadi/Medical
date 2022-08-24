import os
import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras

model = "D:/Work/Lung Tuberculosis/project/InceptionV3 models and ops/seg_tb_collab_initial_update_1.h5"


image_path = 'D:/Work/Lung Tuberculosis/Data sets/Lung Segmentation/normal-5.png'

def pre(path):
    image = cv.imread(path)
    image = cv.resize(image, (256,256))
    image = np.array(image).astype('float32')/255
    image = np.expand_dims(image, axis=0)
    return image

model = keras.models.load_model(model)

image = pre(image_path)
prediction = model.predict(image, verbose = 1)

if prediction > 0.5:
    print('Tuberculosis')
else:
    print('Normal')    

