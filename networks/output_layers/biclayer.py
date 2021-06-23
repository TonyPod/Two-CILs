# -*- coding:utf-8 -*-  

""" 
@time: 2020/5/15 14:47 
@author: Chen He 
@site:  
@file: biclayer.py
@description:  
"""

from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer


class BICLayer(Layer):

    def __init__(self, num_old_classes, **kwargs):
        self.num_old_classes = num_old_classes
        super(BICLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha',
                                     shape=(1),
                                     initializer=Constant(1),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=(1,),
                                    initializer=Constant(0),
                                    trainable=True)

        super(BICLayer, self).build(input_shape)

    def call(self, x):
        # if self.num_old_classes > 0:
        #     alphas = K.concatenate(
        #         (K.ones(self.num_old_classes), K.ones(x.shape[-1] - self.num_old_classes) * self.alpha))
        #     betas = K.concatenate(
        #         (K.zeros(self.num_old_classes), K.ones(x.shape[-1] - self.num_old_classes) * self.beta))
        #     x = x * alphas + betas
        x = K.concatenate((x[:, :self.num_old_classes], x[:, self.num_old_classes:] * self.alpha + self.beta), axis=1)
        return x

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'num_old_classes': self.num_old_classes,
        }
        base_config = super(BICLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
