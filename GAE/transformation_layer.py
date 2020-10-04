#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27-09-2020

@author: tonpoppe
"""

import tensorflow as tf
import numpy as np

class EncTransLayer(tf.keras.layers.Layer):
    """
    Tensorflow layer that reshapes the output of the previous layer into the required format for
    the next encoder. The dimension of the layers are 1) batch size, 2)repetitative part, 3) the
    input size of the encoder.
    """

    def __init__(self, layer_id, support_size, input_shape):
        super(EncTransLayer, self).__init__()

        if layer_id % 2 == 0:
            hub = len(support_size) - int(layer_id / 2)
            if hub == len(support_size) - 1:
                condens = support_size[hub] + hub
            else:
                condens = support_size[hub]
            new_shape = (input_shape[0],
                         int(input_shape[1] / condens),
                         condens * input_shape[2])

        if layer_id % 2 == 1:
            new_shape = (input_shape[0],
                         int(input_shape[1] / 2),
                         2 * input_shape[2])
        self.new_shape = new_shape

    def call(self, input):
        return tf.reshape(input, self.new_shape)

class DecTransLayer(tf.keras.layers.Layer):
    """
    Tensorflow layer that reshapes the output of the previous layer into the required format for
    the next decoder. The dimension of the layers are 1) batch size, 2)repetitative part, 3) the
    input size of the decoder.
    """
    def __init__(self, nw_shape):
        super(DecTransLayer, self).__init__()
        self.nw_shape = nw_shape

    def call(self, input):
        return tf.reshape(input, self.nw_shape)
    
