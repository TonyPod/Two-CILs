# -*- coding:utf-8 -*-  

""" 
@time: 2020/5/31 11:03 
@author: Chen He 
@site:  
@file: output_layer.py
@description:  
"""

import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

from networks.output_layers.biclayer import BICLayer
from networks.output_layers.il2mlayer import IL2MLayer
from networks.output_layers.pslayer import PostScalingLayer


def get_output_layer(args, backbone, class_inc):
    # step 1: get baseline classifier
    if args.baseline == 'fc':
        model = inner_product_output(backbone, len(class_inc.cumul_wnids), weight_decay=args.weight_decay,
                                     use_bias=not args.bias_rect == 'weight_aligning_no_bias')

    else:
        raise Exception('Invalid baseline')

    # step 2: get bias rectification layer
    if not args.base_model:
        if args.bias_rect == 'bic':
            model = bias_correction(model, len(class_inc.old_wnids))
        elif args.bias_rect == 'post_scaling':
            model = post_scaling(model, len(class_inc.cumul_wnids))
        elif args.bias_rect == 'il2m':
            model = il2m_correction(model, len(class_inc.cumul_wnids))

    return model


def inner_product_output(model, num_classes, weight_decay, use_bias=True):
    fc_layer = Dense(num_classes,
                     kernel_regularizer=regularizers.l2(weight_decay),
                     use_bias=use_bias,
                     name='fc')
    return Model(model.input, fc_layer(model.output))


def bias_correction(model, num_old_classes):
    bic_layer = BICLayer(num_old_classes, name='bic_layer')
    return Model(model.input, bic_layer(model.output))


def post_scaling(model, num_classes):
    ps_layer = PostScalingLayer(num_classes, name='ps_layer')
    return Model(model.input, ps_layer(model.output))


def il2m_correction(model, num_classes):
    il2m_layer = IL2MLayer(num_classes, name='il2m_layer')
    return Model(model.input, il2m_layer(model.output))


def special_init(model, old_model, ver='v1'):
    """
    Init the FC parameters of the new class with the average of previous classes'
    :param model: new model
    :param old_model: old model
    :param ver: version
    :return: none
    """
    new_classes_idx = range(old_model.output.shape[-1], model.output.shape[-1])

    if ver == 'v1':

        for layer_idx in range(len(old_model.get_layer('fc').variables)):
            src_tensor = old_model.get_layer('fc').variables[layer_idx].numpy()
            mean_tensor = np.mean(src_tensor, axis=-1)
            for _ in new_classes_idx:
                tmp_tensor = np.expand_dims(mean_tensor, axis=-1)
                tmp_tensor += np.random.normal(0, np.std(src_tensor) / 3.0, tmp_tensor.shape)
                src_tensor = np.concatenate((src_tensor, tmp_tensor), axis=-1)

            # assign new value
            model.get_layer('fc').variables[layer_idx].assign(src_tensor)
        pass
    elif ver == 'v2':
        raise Exception()
    else:
        pass
