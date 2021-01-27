# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:26:26 2021

@author: safayet_khan
"""

# Importing necessary libraries
import os
import splitfolders


# Necessary constant values are being fixed
RANDOM_SEED = 1337
RATIO = (0.8, 0.2) # Train set and validation set ratio
SRC_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/processed-data/train'
DST_DIR = 'C:/Users/safayet_khan/Desktop/plant-final/final-data'


# Creating destination folder if it's not exist already
if not os.path.exists(path=DST_DIR):
    os.mkdir(path=DST_DIR)
os.chdir(path=DST_DIR)


# Random shuffling train set and validation set with the fixed ratio
splitfolders.ratio(SRC_DIR, output=DST_DIR, seed=RANDOM_SEED,
                   ratio=RATIO, group_prefix=None)
