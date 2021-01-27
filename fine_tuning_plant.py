# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:45:59 2021

@author: safayet_khan
"""

# This notebook demonstrates training an [Inception]
# (https://arxiv.org/abs/1512.00567) model with pre-trained ImageNet
# weights for the ["Plant Seedlings Classification"]
# (https://www.kaggle.com/c/plant-seedlings-classification) contest.
# I fine-tuned the previous model (plant_seedlings_TL_WO_fine_tuning)
# for this purpose. The whole model will be trained with a very
# small learning rate.


# Importing necessary libraries
import math
import os
import glob
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler, TerminateOnNaN
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
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
TEST_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/final-data/test'


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


# Load the previous model which was without fine-tuning
MODEL_SOURCE = 'C:/Users/safayet_khan/Desktop/plant-final/checkpoint_wo_ft'
MODEL_PATH = os.path.join(MODEL_SOURCE, 'plant_model.h5')

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    model.summary()


# Unfreezing the weights of Inception model
for layer in model.layers:
    layer.trainable = True
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
CHECKPOINT_PATH = 'C:/Users/safayet_khan/Desktop/plant-final/checkpoint_ft'
if not os.path.exists(path=CHECKPOINT_PATH):
    os.mkdir(path=CHECKPOINT_PATH)

MODEL_FILEPATH = os.path.join(CHECKPOINT_PATH, 'plant_model.h5')
checkpoint_callback = ModelCheckpoint(filepath=MODEL_FILEPATH,
                                      monitor='val_loss', mode='min',
                                      verbose=1, save_weights_only=False,
                                      save_best_only=True, save_freq='epoch')


# LearningRateScheduler Callback
INITIAL_LEARNING_RATE = 0.00001

def lr_step_decay(epoch):
    '''
    Reduce the learning rate by a certain percentage after a certain
    number of epochs.
    '''
    drop_rate = 0.25
    epochs_drop = 10.0
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
NUMBER_OF_EPOCHS = 100


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


# Loading Test images paths
test_image_paths = glob.glob(os.path.join(TEST_DIR, '*.png'))
print(np.shape(test_image_paths))


# Loading the Test images in an array. Resizing and
# Rescaling is also being done
x_test = np.empty((np.shape(test_image_paths)[0], IMAGE_SIZE_3D[0],
                   IMAGE_SIZE_3D[1], IMAGE_SIZE_3D[2]), dtype=np.uint8)
test_file_names = []

for i, file_path in enumerate(test_image_paths):
    file_name = file_path.split(sep=os.path.sep)[-1]
    test_file_names.append(file_name)
    img_array = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, IMAGE_SIZE_2D)
    x_test[i, ] = img_array
x_test = preprocess_input(x_test)

print(type(x_test))
print(x_test.shape)

# Loading the best model after fine-tuning
BEST_MODEL_PATH = os.path.join(CHECKPOINT_PATH, 'plant_model.h5')
if os.path.exists(MODEL_PATH):
    best_model = load_model(BEST_MODEL_PATH)
    best_model.summary()


# Making prediction with the model
prediction_probabilities = best_model.predict(x_test)

print(type(prediction_probabilities))
print(np.shape(prediction_probabilities))
print(prediction_probabilities[5])


# Extracting predicted labels from the prediction
prediction_labels = np.argmax(prediction_probabilities, axis=1)

print(type(prediction_labels))
print(np.shape(prediction_labels))
print(prediction_labels[5])


# Converting predicted labels to Class name
LABELS_TO_CLASS = {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers',
                   3: 'Common Chickweed', 4: 'Common wheat', 5: 'Fat Hen',
                   6: 'Loose Silky-bent', 7: 'Maize', 8: 'Scentless Mayweed',
                   9: 'Shepherds Purse', 10: 'Small-flowered Cranesbill',
                   11: 'Sugar beet'}
prediction_string = []

for i in prediction_labels:
    prediction_string.append(LABELS_TO_CLASS[i])

print(type(prediction_string))
print(np.shape(prediction_string))
print(prediction_string[0])


# Converting the results to a CSV file
result = {'file': test_file_names, 'species': prediction_string}
result = pd.DataFrame(result)
result.to_csv("Plant_Seedlings.csv", index=False)
