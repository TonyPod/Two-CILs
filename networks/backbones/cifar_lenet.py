# -*- coding:utf-8 -*-  

""" 
@time: 12/4/19 11:38 PM 
@author: Chen He 
@site:  
@file: cifar_net.py
@description:  
"""

from tensorflow.keras import Model, backend, layers, regularizers
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Input


def lenet(weight_decay=1e-4, final_relu=True):
    input_shape = (32, 32, 3)
    img_input = Input(shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        img_input = layers.Lambda(
            lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
            name='transpose')(img_input)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    x = img_input

    x = Conv2D(32, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = MaxPool2D()(x)

    x = Conv2D(64, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu' if final_relu else None, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Dropout(0.5)(x)

    return Model(img_input, x, name='lenet')
