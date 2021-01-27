# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 12:15:57 2021

@author: safayet_khan
"""

# This script extracts the green color of the leaves and saves the processed
# images in another directory. I wrote this script, for the "Plant Seedling
# Classification" contest, to remove the background noise from the images.


# Importing necessary libraries
import os
import glob
import cv2
import matplotlib.image as mpimg


# Necessary constant values are being fixed
IMAGE_SIZE = (299, 299)
DARK_GREEN = (25, 40, 40) # Most dark green in RGB
LIGHT_GREEN = (80, 255, 255) # Most light green in RGB
# Replace all the pixels excluding DARK_GREEN and LIGHT_GREEN to black pixels
CHANNEL_VALUE = [0, 0, 0]

FOLDER_NAMES = ['train', 'test']
PLANT_CLASSES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
                 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize',
                 'Scentless Mayweed', 'Shepherds Purse',
                 'Small-flowered Cranesbill', 'Sugar beet']

# MAIN_DIR contains the original images
MAIN_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/plant-seedlings'
# DATA_DIR contains the processed images
DATA_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/processed-data'

# Creating processed-data folder
if not os.path.exists(path=DATA_DIR):
    os.mkdir(path=DATA_DIR)


for folder_name in FOLDER_NAMES:
    read_dir = os.path.join(MAIN_DIR, folder_name)
    write_dir = os.path.join(DATA_DIR, folder_name)

    if not os.path.exists(path=write_dir):
        os.mkdir(path=write_dir)
    os.chdir(path=write_dir)

    if folder_name=='train':
        for plant_class in PLANT_CLASSES:
            path_images = glob.glob(os.path.join(read_dir, plant_class,
                                                 '*.png'))
            class_dir = os.path.join(write_dir, plant_class)
            # Creating folder for each of the PLANT_CLASSES
            if not os.path.exists(path=class_dir):
                os.mkdir(path=class_dir)
            os.chdir(path=class_dir)

            for path_image in path_images:
                img_array = cv2.imread(path_image, cv2.IMREAD_COLOR)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                img_array = cv2.resize(img_array, IMAGE_SIZE)

                # Converting the RGB values to HSV values
                hsv_img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv_img_array)
                mask = cv2.inRange(hsv_img_array, DARK_GREEN, LIGHT_GREEN)
                img_array[mask<255] = CHANNEL_VALUE
                FILE_NAME = '{}.png'.format(path_image[-13:-4])
                mpimg.imsave(FILE_NAME, img_array)

    else:
        path_images = glob.glob(os.path.join(read_dir, '*.png'))
        if not os.path.exists(path=write_dir):
            os.mkdir(path=write_dir)
        os.chdir(path=write_dir)

        for path_image in path_images:
            img_array = cv2.imread(path_image, cv2.IMREAD_COLOR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, IMAGE_SIZE)

            # Converting the RGB values to HSV values
            hsv_img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv_img_array)
            mask = cv2.inRange(hsv_img_array, DARK_GREEN, LIGHT_GREEN)
            img_array[mask<255] = CHANNEL_VALUE
            FILE_NAME = '{}.png'.format(path_image[-13:-4])
            mpimg.imsave(FILE_NAME, img_array)
