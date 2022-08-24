import os
import cv2 as cv
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import normalize
from tqdm import tqdm



image_directory = 'D:/Work/Lung Tuberculosis/Data sets/Lung Segmentation/training data/'
mask_directory = 'D:/Work/Lung Tuberculosis/Data sets/Lung Segmentation/agumented/masks/'
SIZE = 256



image_dataset = []  
mask_dataset = []  

images = os.listdir(image_directory)
for i, image_name in tqdm(enumerate(images)):    
    if (image_name.split('.')[1] == 'png'):
        #print(image_directory+image_name)
        image = cv.imread(image_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in tqdm(enumerate(masks)):
    if (image_name.split('.')[1] == 'png'):
        image = cv.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

image_dataset_uint8=np.array(image_dataset)
mask_dataset_uint8=np.array(mask_dataset)

#print("Used memory to store the 8 bit int image dataset is: ", image_dataset_uint8.nbytes/(1024*1024), "MB")
#print("Used memory to store the 8 bit int mask dataset is: ", mask_dataset_uint8.nbytes/(1024*1024), "MB")


image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

#print("Used memory to store the float image dataset is: ", image_dataset.nbytes/(1024*1024), "MB")
#print("Used memory to store the float mask dataset is: ", mask_dataset.nbytes/(1024*1024), "MB")


from sklearn.model_selection import train_test_split
image_train, image_test, mask_train, mask_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

#print(image_dataset.shape)
print(len(image_train))
print(len(image_test))
