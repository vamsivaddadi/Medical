import numpy as np
import cv2 as cv
from glob import glob
from Unet import *
from cup_data_seg import *
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)

batch_size = 5
model = unet()

callbacks = [
        EarlyStopping(patience=3, monitor='val_loss', mode = 'min'),
        TensorBoard(log_dir='logs')]


model_checkpoint = ModelCheckpoint('cup_unet_checkpoints.hdf5',
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
                        #callbacks=callbacks
                             )


model.save('models/full_seg_cup_unet1.h5')

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
#plt.savefig('cup Training and validation loss1.png')

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
#plt.savefig('disc Training and validation accuracy.png')



