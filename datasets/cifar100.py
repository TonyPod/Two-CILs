# -*- coding:utf-8 -*-

########################################################################
#
# Functions for downloading the CIFAR-100 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import pickle

import numpy as np
import tensorflow as tf

from datasets.dataset import Dataset
from datasets.datautils import change_label_order, create_subset_version

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
# data_path = os.path.join(os.path.dirname(__file__), 'cifar100')
data_path = '/home/hechen/Datasets/CIFAR-100'

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
IMG_WIDTH = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = 3

# Length of an image when flattened to a 1-dim array.
IMG_SIZE = IMG_WIDTH * IMG_WIDTH * NUM_CHANNELS

# Number of classes.
# PIXEL_MEAN_FILE = os.path.join(data_path, 'mean.npy')
PIXEL_MEAN_FILE = os.path.join(os.path.dirname(__file__), 'cifar100/mean.npy')
if os.path.exists(PIXEL_MEAN_FILE):
    pixel_mean = np.load(PIXEL_MEAN_FILE)
else:
    raise Exception('CIFAR-100 mean file not found!')


########################################################################
# Various constants used to allocate arrays of the correct size.


########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    import sys
    if sys.version_info <= (3, 3):
        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file)
    else:
        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw, mean_subtraction=True):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = raw.astype(np.float32) / 255.0

    # Reshape the array to 4-dimensions.
    if raw_float.ndim == 2:
        images = raw_float.reshape([-1, NUM_CHANNELS, IMG_WIDTH, IMG_WIDTH])
    else:
        images = raw_float

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    if mean_subtraction:
        images = images - pixel_mean

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'fine_labels'], dtype=np.int32)
    coarse_cls = np.array(data[b'coarse_labels'], dtype=np.int32)

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls, raw_images, coarse_cls


class CIFAR100(Dataset):
    NUM_CLASSES = 100
    NUM_MAX_TRAIN_SAMPLES_PER_CLASS = 500
    NUM_MAX_TEST_SAMPLES_PER_CLASS = 100

    def __init__(self, args):
        self.args = args

        # Load the class-names from the pickled file.
        raw = _unpickle(filename="meta")[b'fine_label_names']

        # Convert from binary strings.
        names = [x.decode('utf-8') for x in raw]
        self.label_names_dict = {i: names[i] for i in range(len(names))}

        print('Loading cifar100 data...')
        self.train_images, self.train_cls, self.train_raw_images, _ = _load_data(filename='train')
        self.test_images, self.test_cls, self.test_raw_images, _ = _load_data(filename='test')

        print('Loading complete')

    def aug_fn(self, image, label):
        image = tf.image.pad_to_bounding_box(image, 3, 3, 42, 42)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
        return image, label

    def preprocess(self, images, **kwargs):
        images = _convert_images(images, **kwargs)
        return images

    def load_train(self, cur, order, num_samples):
        """
        Load all the training-data for the CIFAR-10 data-set.
        The data-set is split into 5 data-files which are merged here.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        selected_indices = [i for i, l in enumerate(self.train_cls) if l in cur]
        images = self.train_images[selected_indices]
        cls = self.train_cls[selected_indices]

        cls = change_label_order(cls, order)

        if not num_samples == -1:
            images, cls = create_subset_version(images, cls, [num_samples] * self.NUM_CLASSES)

        dataset = tf.data.Dataset.from_tensor_slices((images, cls))

        return dataset

    def load_test(self, cur, order):
        """
        Load all the test-data for the CIFAR-10 data-set.
        Returns the images, class-numbers and one-hot encoded class-labels.
        """

        selected_indices = [i for i, l in enumerate(self.test_cls) if l in cur]
        images = self.test_images[selected_indices]
        cls = self.test_cls[selected_indices]

        cls = change_label_order(cls, order)

        dataset = tf.data.Dataset.from_tensor_slices((images, cls))

        return dataset
