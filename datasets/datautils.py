# -*- coding:utf-8 -*-  

""" 
@time: 12/5/19 10:40 PM 
@author: Chen He 
@site:  
@file: utils_func.py
@description:  
"""

import copy

import numpy as np


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def change_label_order(cls, order):
    '''
    Change the labels according to the given order to make it agree with the indices of the FC output
    :param cls: labels
    :param order: class order
    :return: re-ordered class order
    '''
    order_dict = dict()
    for i in range(len(order)):
        order_dict[order[i]] = i

    reordered_cls = np.array([order_dict[cls[i]] for i in range(len(cls))], dtype=np.int32)
    return reordered_cls


def create_subset_version(images, cls, num_samples_per_class):
    selection_mask = np.zeros(len(cls))
    samples_count = copy.deepcopy(num_samples_per_class)
    for i in range(len(cls)):
        cur_label = cls[i]
        if samples_count[cur_label] > 0:
            selection_mask[i] = 1
            samples_count[cur_label] -= 1
    selection_mask = selection_mask.astype(np.bool)

    if isinstance(images, list):
        images = [images[i] for i, status in enumerate(selection_mask) if status == True]
    else:
        images = images[selection_mask]

    if isinstance(cls, list):
        cls = [cls[i] for i, status in enumerate(selection_mask) if status == True]
    else:
        cls = cls[selection_mask]
    return images, cls
