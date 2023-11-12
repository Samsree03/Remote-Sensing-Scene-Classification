import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import keras
from keras import models
from keras.models import Sequential, Model,load_model

train_path=r"C:/Users/sams.sridharan/Downloads/SAM/SAM/train" 
train_ds= tf.keras.utils.image_dataset_from_directory('C:/Users/sams.sridharan/Downloads/SAM/SAM/train')
valid_path=r"C:/Users/sams.sridharan/Downloads/SAM/SAM/valid"
val_ds= tf.keras.utils.image_dataset_from_directory('C:/Users/sams.sridharan/Downloads/SAM/SAM/valid')

class_names = train_ds.class_names
print(class_names)

mobilenet = Sequential()

pretrained_model= tf.keras.applications.MobileNetV2(include_top=False,
                   input_shape=(256,256,3),
                   pooling='avg',classes=45,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

mobilenet.add(pretrained_model)
mobilenet.add(Flatten())

mobilenet.add(Dense(512, activation='relu'))
mobilenet.add(Dense(256, activation='relu'))
mobilenet.add(Dense(128, activation='relu'))
mobilenet.add(Dense(64, activation='relu'))
mobilenet.add(Dense(32, activation='relu'))
mobilenet.add(Dense(45, activation='softmax'))

mobilenet.summary()

mobilenet.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
'''mobilenet.compile(optimizer=Adam(),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),model_metrics=['acc'])'''

import scipy
history = mobilenet.fit(train_ds.repeat(), 
                              batch_size=64,
                              steps_per_epoch=15, 
                              epochs=400,
                              validation_data=val_ds.repeat(), 
                              validation_steps=15, 
                              verbose=1)

file_name = input("Enter the file name to be saved :)")
mobilenet.save_weights("%s.h5" % file_name)

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.savefig('mobilenet 15spe 400epoch acc')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.savefig('mobile 15spe 400epoch loss')
plt.show()
