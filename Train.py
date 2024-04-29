import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import os
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Modified import statement
from tensorflow.keras.applications import ResNet50  # Modified import statement
# import matplotlib.pyplot as plt
# %matplotlib inline

train_dir='data/train'
test_dir='data/test'

def lr_schedule(epoch):
    initial_lr = 0.0001
    decay_factor = 0.1
    decay_epochs = 10
    return initial_lr * (decay_factor ** (epoch // decay_epochs))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1.0/255.0,
                                 rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True)
val_datagen=ImageDataGenerator(rescale=1.0/255.0)

trainDatagen=train_datagen.flow_from_directory(train_dir,
                                               target_size=(48,48),
                                               batch_size=100,
                                               class_mode='categorical',
                                               color_mode='rgb')

valDatagen=val_datagen.flow_from_directory(test_dir,
                                           target_size=(48,48),
                                           batch_size=100,
                                           class_mode='categorical',
                                           color_mode='rgb')

base_model=ResNet50(input_shape=(48,48,3),include_top=False,weights='imagenet')

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
model=Sequential()
model.add(base_model)
model.add(Conv2D(filters=32,kernel_size=3,activation='relu',padding='same'))

model.add(Conv2D(filters=64,kernel_size=3,activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=3,activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=3,padding='same'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

history=model.fit(trainDatagen,validation_data=valDatagen,epochs=5,callbacks=[lr_scheduler])

# model_json=model.to_json()
# with open("emotion_model_resnet.json","w") as json_file:
#     json_file.write(model_json)

model.save_weights('emotion_model_resnet.weights.h5')