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
import cv2

train_path=r"C:/Users/sams.sridharan/Downloads/SAM/SAM/train" 
train_ds= tf.keras.utils.image_dataset_from_directory('C:/Users/sams.sridharan/Downloads/SAM/SAM/train')
valid_path=r"C:/Users/sams.sridharan/Downloads/SAM/SAM/valid"
val_ds= tf.keras.utils.image_dataset_from_directory('C:/Users/sams.sridharan/Downloads/SAM/SAM/valid')

class_names = train_ds.class_names
print(class_names)

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(256,256,3),
                   pooling='avg',classes=3,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dense(64, activation='relu'))
resnet_model.add(Dense(32, activation='relu'))
resnet_model.add(Dense(45, activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

import scipy
history = resnet_model.fit(train_ds.repeat(), 
                              batch_size=62,
                              steps_per_epoch=10, 
                              epochs=15,
                              validation_data=val_ds.repeat(), 
                              validation_steps=15, 
                              verbose=1)

pic="C:/Users/sams.sridharan/Downloads/UCMerced_LandUse/Images/beach/beach09.tif"
image=cv2.imread(pic)
image_resized= cv2.resize(image, (256,256))
image=np.expand_dims(image_resized,axis=0)
pred=resnet_model.predict(image)
print(pred)
output_class=class_names[np.argmax(pred)]
print("image is",output_class)


file_name = input("Enter the file name to be saved :)")
resnet_model.save("%s.h5" % file_name)

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.savefig('resnet 15spe 400epoch acc')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.savefig('resnet 15spe 400epoch loss')
plt.show()
