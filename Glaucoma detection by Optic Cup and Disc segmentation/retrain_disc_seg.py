import numpy as np
import cv2 as cv
from glob import glob
from Unet import *
from disc_data_seg import *
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import random
import tensorflow as tf
from keras.models import load_model

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

batch_size = 5
model = load_model('models/retrain_full_t1_ha_disc.h5')

callbacks = [
        EarlyStopping(patience=3, monitor='val_loss', mode = 'min'),
        TensorBoard(log_dir='logs')]


model_checkpoint = ModelCheckpoint('disc_unet_checkpoints.hdf5',
                                    monitor = 'val_accuracy',
                                    verbose = 1,
                                    save_best_only= True,
                                    mode= 'max')


history = model.fit(image_train,
                        mask_train,
                        batch_size= batch_size,
                        steps_per_epoch= np.ceil(len(image_train)//batch_size),
                        epochs = 10, 
                        validation_steps= np.ceil(len(image_test)//batch_size),     
                        validation_data = (image_test, mask_test),
                        
                             )


model.save('models/retrain_full_noise_t1_ha_disc.h5')

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#plt.savefig('Training and validation loss3.png')

acc = history.history['accuracy']
#print('Accuracy = ', acc)
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#print('Val_Accuracy',val_acc)
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#plt.savefig('Training and validation accuracy3.png')



