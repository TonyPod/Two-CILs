# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers

from utils import imagenet_preprocessing

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def basic_block(input_tensor,
                filters,
                stage,
                block,
                strides=(1, 1),
                shortcut=False,
                weight_decay=1e-4,
                final_relu=True):
    """
    Basic block for ResNet18
    Args:
        input_tensor:
        filters:
        stage:
        block:

    Returns:

    """
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filters1, (3, 3),
        padding='same',
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2a')(
        input_tensor)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2a')(
        x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, (3, 3),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2b')(
        x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2b')(
        x)

    if shortcut:
        y = Conv2D(
            filters2, (1, 1),
            strides=strides,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(weight_decay),
            name=conv_name_base + '1')(
            input_tensor)
        y = BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            name=bn_name_base + '1')(
            y)
        x = layers.add([x, y])
    else:
        x = layers.add([x, input_tensor])

    if final_relu:
        x = Activation('relu')(x)

    return x


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   weight_decay=1e-4):
    """The identity block is the block that has no conv layer at shortcut.

    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.

    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2a')(
        input_tensor)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2a')(
        x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2b')(
        x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2b')(
        x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2c')(
        x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2c')(
        x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               weight_decay=1e-4):
    """A block that has a conv layer at shortcut.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well

    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.

    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2a')(
        input_tensor)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2a')(
        x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2b')(
        x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2b')(
        x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '2c')(
        x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '2c')(
        x)

    shortcut = Conv2D(
        filters3, (1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name=conv_name_base + '1')(
        input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(
        shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


#
# def resnet(weight_decay=1e-4,
#            rescale_inputs=False):
#     """Instantiates the ResNet50 architecture.
#
#     Args:
#       use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
#       rescale_inputs: whether to rescale inputs from 0 to 1.
#
#     Returns:
#         A Keras model instance.
#     """
#     input_shape = (64, 64, 3)
#     img_input = layers.Input(shape=input_shape)
#     if rescale_inputs:
#         # Hub image modules expect inputs in the range [0, 1]. This rescales these
#         # inputs to the range expected by the trained model.
#         x = layers.Lambda(
#             lambda x: x * 255.0 - backend.constant(
#                 imagenet_preprocessing.CHANNEL_MEANS,
#                 shape=[1, 1, 3],
#                 dtype=x.dtype),
#             name='rescale')(
#             img_input)
#     else:
#         x = img_input
#
#     if backend.image_data_format() == 'channels_first':
#         x = layers.Lambda(
#             lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
#             name='transpose')(x)
#         bn_axis = 1
#     else:  # channels_last
#         bn_axis = 3
#
#     x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
#     x = Conv2D(
#         64, (3, 3),
#         strides=(2, 2),
#         padding='valid',
#         use_bias=False,
#         kernel_initializer='he_normal',
#         kernel_regularizer=regularizers.l2(weight_decay),
#         name='conv1')(
#         x)
#     x = BatchNormalization(
#         axis=bn_axis,
#         momentum=BATCH_NORM_DECAY,
#         epsilon=BATCH_NORM_EPSILON,
#         name='bn_conv1')(
#         x)
#     x = Activation('relu')(x)
#
#     x = conv_block(
#         x,
#         3, [32, 32, 128],
#         stage=2,
#         block='a',
#         strides=(1, 1),
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [32, 32, 128],
#         stage=2,
#         block='b',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [32, 32, 128],
#         stage=2,
#         block='c',
#         weight_decay=weight_decay)
#
#     x = conv_block(
#         x,
#         3, [64, 64, 256],
#         stage=3,
#         block='a',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [64, 64, 256],
#         stage=3,
#         block='b',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [64, 64, 256],
#         stage=3,
#         block='c',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [64, 64, 256],
#         stage=3,
#         block='d',
#         weight_decay=weight_decay)
#
#     x = conv_block(
#         x,
#         3, [128, 128, 512],
#         stage=4,
#         block='a',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [128, 128, 512],
#         stage=4,
#         block='b',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [128, 128, 512],
#         stage=4,
#         block='c',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [128, 128, 512],
#         stage=4,
#         block='d',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [128, 128, 512],
#         stage=4,
#         block='e',
#         weight_decay=weight_decay)
#     x = identity_block(
#         x,
#         3, [128, 128, 512],
#         stage=4,
#         block='f',
#         weight_decay=weight_decay)
#
#     rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
#     x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
#
#     # Create model.
#     return Model(img_input, x, name='resnet')


def resnet18(weight_decay=1e-4,
             rescale_inputs=False,
             final_relu=True):
    """Instantiates the ResNet50 architecture.

    Args:
      use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
      rescale_inputs: whether to rescale inputs from 0 to 1.

    Returns:
        A Keras model instance.
    """
    input_shape = (64, 64, 3)
    img_input = layers.Input(shape=input_shape)
    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        x = layers.Lambda(
            lambda x: x * 255.0 - backend.constant(
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=x.dtype),
            name='rescale')(
            img_input)
    else:
        x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Lambda(
            lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
            name='transpose')(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    x = Conv2D(
        64, (3, 3),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(weight_decay),
        name='conv1')(
        x)
    x = BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name='bn_conv1')(
        x)
    x = Activation('relu')(x)

    x = basic_block(
        x, [32, 32],
        stage=2,
        block='a',
        weight_decay=weight_decay,
        shortcut=True)
    x = basic_block(
        x, [32, 32],
        stage=2,
        block='b',
        weight_decay=weight_decay)

    x = basic_block(
        x, [64, 64],
        strides=(2, 2),
        stage=3,
        block='a',
        weight_decay=weight_decay,
        shortcut=True)
    x = basic_block(
        x, [64, 64],
        stage=3,
        block='b',
        weight_decay=weight_decay)

    x = basic_block(
        x, [128, 128],
        strides=(2, 2),
        stage=4,
        block='a',
        weight_decay=weight_decay,
        shortcut=True)
    x = basic_block(
        x, [128, 128],
        stage=4,
        block='b',
        weight_decay=weight_decay)

    x = basic_block(
        x, [256, 256],
        strides=(2, 2),
        stage=5,
        block='a',
        weight_decay=weight_decay,
        shortcut=True)
    x = basic_block(
        x, [256, 256],
        stage=5,
        block='b',
        weight_decay=weight_decay,
        final_relu=final_relu)

    rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)

    # Create model.
    return Model(img_input, x, name='resnet18')
