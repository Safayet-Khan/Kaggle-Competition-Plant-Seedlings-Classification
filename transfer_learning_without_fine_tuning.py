# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:09:57 2021

@author: safayet_khan
"""

# This script demonstrates training an
# [Inception](https://arxiv.org/abs/1512.00567) model with pre-trained
# ImageNet weights to classify the ["Plant Seedlings Classification"]
# (https://www.kaggle.com/c/plant-seedlings-classification) contest.
# Transfer learning without fine-tuning is used to train the layers on
# top of the Inception model. In the next script, the whole model will
# be trained with a small learning rate for fine-tuning purposes.


# Importing necessary libraries
import math
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
tf.config.experimental_run_functions_eagerly(True)


# Image size, batch size, and other necessary values are being fixed
RANDOM_SEED = 42
BATCH_SIZE = 64
IMAGE_SIZE = 299
CHANNEL_NUMBER = 3
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUMBER)
CLASSES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
           'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize',
           'Scentless Mayweed', 'Shepherds Purse',
           'Small-flowered Cranesbill', 'Sugar beet']
TRAIN_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/final-data/train'
VALID_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/final-data/val'


# Rescaling and creating augmentation with ImageDataGenerator for
# training set and validation set
train_data = ImageDataGenerator(rotation_range=180,
                                brightness_range=[0.9, 1.1],
                                preprocessing_function=preprocess_input,
                                fill_mode='constant',
                                cval=0.0)
valid_data = ImageDataGenerator(preprocessing_function=preprocess_input)


# Data generator for the training set
train_generator = train_data.flow_from_directory(directory=TRAIN_DIR,
                                                 target_size=IMAGE_SIZE_2D,
                                                 color_mode='rgb',
                                                 classes=CLASSES,
                                                 class_mode='categorical',
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 seed=RANDOM_SEED)
print(train_generator.class_indices)


# Data generator for the validation set
valid_generator = valid_data.flow_from_directory(directory=VALID_DIR,
                                                 target_size=IMAGE_SIZE_2D,
                                                 color_mode='rgb',
                                                 classes=CLASSES,
                                                 class_mode='categorical',
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 seed=RANDOM_SEED)
print(valid_generator.class_indices)


# Loading Inception model with pre-trained ImageNet weights
base_model = InceptionV3(include_top=False, weights='imagenet',
                         input_tensor=None, input_shape=IMAGE_SIZE_3D,
                         pooling='avg')
base_model.summary()


# Adding layers on top of Inception model
top_model = Dense(units=1024, activation='relu')(base_model.output)
top_model = Dropout(rate=0.3)(top_model)
top_model = Dense(units=1024, activation='relu')(top_model)
top_model = Dropout(rate=0.3)(top_model)
top_model = Dense(units=len(CLASSES), activation='softmax')(top_model)

model = tf.keras.Model(inputs=base_model.input, outputs=top_model)
model.summary()


# Freezing the weights of Inception model
for layer in base_model.layers:
    layer.trainable = False
model.summary()


# Callback (Stopped after reaching a certain value)
VALIDATION_ACCURACY_THRESHOLD = 0.9999

class MyCallback(Callback):
    '''
    Stop training after val_accuracy reached a certain number.
    '''
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy')>VALIDATION_ACCURACY_THRESHOLD:
            print('\nCOMPLETED!!!')
            self.model.stop_training=True

callbacks = MyCallback()


# ModelCheckpoint Callback
CHECKPOINT_PATH = 'C:/Users/safayet_khan/Desktop/plant-final/checkpoint_wo_ft'
if not os.path.exists(path=CHECKPOINT_PATH):
    os.mkdir(path=CHECKPOINT_PATH)

MODEL_FILEPATH = os.path.join(CHECKPOINT_PATH, 'plant_model.h5')
checkpoint_callback = ModelCheckpoint(filepath=MODEL_FILEPATH,
                                      monitor='val_loss', mode='min',
                                      verbose=1, save_weights_only=False,
                                      save_best_only=True, save_freq='epoch')


# LearningRateScheduler Callback
INITIAL_LEARNING_RATE = 0.001

def lr_step_decay(epoch):
    '''
    Reduce the learning rate by a certain percentage after a certain
    number of epochs.
    '''
    drop_rate = 0.15
    epochs_drop = 5.0
    return INITIAL_LEARNING_RATE * math.pow((1-drop_rate),
                                            math.floor(epoch/epochs_drop))

lr_callback = LearningRateScheduler(schedule=lr_step_decay, verbose=1)


# CSVLogger Callback
CSVLOGGER_FILEPATH = os.path.join(CHECKPOINT_PATH, 'plant_log.csv')
CSVLOGGER_callback = CSVLogger(filename=CSVLOGGER_FILEPATH, separator=',',
                               append=False)


# TerminateOnNaN Callback
Terminate_callback = TerminateOnNaN()


# Step per epoch, Validation steps, and number of Epochs to be trained
STEPS_PER_EPOCH = math.ceil(train_generator.samples/BATCH_SIZE)
VALIDATION_STEPS = math.ceil(valid_generator.samples/BATCH_SIZE)
NUMBER_OF_EPOCHS = 30


# Compiling the Model
EPSILON = 0.1

model.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE,
                             epsilon=EPSILON),
              loss='categorical_crossentropy', metrics=['accuracy'])


# Training the Model
model.fit(train_generator, shuffle=True, epochs=NUMBER_OF_EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH, verbose=1,
          validation_data=valid_generator, validation_steps=VALIDATION_STEPS,
          callbacks=[callbacks, checkpoint_callback, lr_callback,
                     CSVLOGGER_callback, Terminate_callback])
