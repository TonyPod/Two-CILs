# -*- coding:utf-8 -*-  

""" 
@time: 3/3/21 10:05 PM 
@author: Chen He 
@site:  
@file: imagenet64x64_mobilenetv2.py
@description:  
"""

from tensorflow.python.keras import backend, Model, layers, regularizers


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenet_64x64(weight_decay=1e-4,
                    final_relu=True,
                    alpha=1.0,
                    pooling='avg'):
    """Instantiates the MobileNetV2 architecture.
    Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)
    Optionally loads weights pre-trained on ImageNet.
    Note: each Keras Application expects a specific kind of input preprocessing.
    For MobileNetV2, call `tf.keras.applications.mobilenet_v2.preprocess_input`
    on your inputs before passing them to the model.
    Arguments:
      alpha: Float between 0 and 1. controls the width of the network.
        This is known as the width multiplier in the MobileNetV2 paper,
        but the name is kept for consistency with `applications.MobileNetV1`
        model in Keras.
        - If `alpha` < 1.0, proportionally decreases the number
            of filters in each layer.
        - If `alpha` > 1.0, proportionally increases the number
            of filters in each layer.
        - If `alpha` = 1, default number of filters from the paper
            are used at each layer.
      pooling: String, optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model
            will be the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a
            2D tensor.
        - `max` means that global max pooling will
            be applied.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape or invalid alpha, rows when
        weights='imagenet'
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """

    img_input = layers.Input(shape=(64, 64, 3))

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name='Conv1')(img_input)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(
        x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(
        x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, weight_decay=weight_decay)

    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, weight_decay=weight_decay)

    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, weight_decay=weight_decay)

    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9, weight_decay=weight_decay)

    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12, weight_decay=weight_decay)

    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14, weight_decay=weight_decay)
    x = _inverted_res_block(
        x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15, weight_decay=weight_decay)

    x = _inverted_res_block(
        x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16, weight_decay=weight_decay)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name='Conv_1',
        kernel_regularizer=regularizers.l2(weight_decay))(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(
        x)
    if final_relu:
        x = layers.ReLU(6., name='out_relu')(x)

    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    else:
        raise Exception()

    x = layers.Flatten()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenetv2_%0.2f' % alpha)

    return model


def _correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Arguments:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, weight_decay):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand_BN')(
            x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=_correct_pad(x, 3),
            name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        padding='same' if stride == 1 else 'valid',
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise_BN')(
        x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        activation=None,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project_BN')(
        x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x
