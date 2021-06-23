# -*- coding:utf-8 -*-  

""" 
@time: 2020/5/15 14:47 
@author: Chen He 
@site:  
@file: pslayer.py
@description:  
"""

from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer


class PostScalingLayer(Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(PostScalingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.logits_bias = self.add_weight(name='logits_bias',
                                           shape=(self.num_classes,),
                                           initializer=Constant(0),
                                           trainable=False)

        super(PostScalingLayer, self).build(input_shape)

    def call(self, x):
        x += self.logits_bias
        return x

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'num_classes': self.num_classes,
        }
        base_config = super(PostScalingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
