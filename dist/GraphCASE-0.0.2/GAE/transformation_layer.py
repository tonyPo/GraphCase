#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27-09-2020

@author: tonpoppe
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
import numpy as np

class EncTransLayer(tf.keras.layers.Layer):
    """
    Tensorflow layer that reshapes the output of the previous layer into the required format for
    the next encoder. The dimension of the layers are 1) batch size, 2)repetitative part, 3) the
    input size of the encoder.

    args:
        layer_id    : number of the target layer with count starting at 1
    """

    def __init__(self, layer_id, support_size, new_shape=None, **kwargs):
        super(EncTransLayer, self).__init__(**kwargs)
        self.layer_id = layer_id
        self.support_size = support_size
        self.new_shape = new_shape
    
    @staticmethod
    def get_output_dim(layer_id, support_size, dims, feature_dim):
        if layer_id == 1:  # original feature layer
            return feature_dim

        if layer_id % 2 == 0:  # combine hub
            hub = len(support_size) - int(layer_id / 2)
            if hub == len(support_size) - 1:
                condens = support_size[hub] + hub
            else:
                condens = support_size[hub]
            return condens * dims[layer_id-2]

        if layer_id % 2 != 0:  # combine in + out neightboorhood
            return 2 * dims[layer_id-2]

    def build(self, input_shape):
        if self.layer_id % 2 == 0:
            hub = len(self.support_size) - int(self.layer_id / 2)
            if hub == len(self.support_size) - 1:
                condens = self.support_size[hub] + hub
            else:
                condens = self.support_size[hub]
            new_shape = (input_shape[0],
                         int(input_shape[1] / condens),
                         condens * input_shape[2])

        if self.layer_id % 2 == 1:
            new_shape = (input_shape[0],
                         int(input_shape[1] / 2),
                         2 * input_shape[2])
        self.new_shape = new_shape
        # print(f"{self.name} input: {input_shape} output:{self.new_shape.numpy()}")

    def call(self, inputs):
        target_shape = (
            tf.shape(inputs)[0],
            self.new_shape[1],
            self.new_shape[2])
        return tf.reshape(inputs, target_shape)

    def get_config(self):
        config = super(EncTransLayer, self).get_config()
        config.update({
            "layer_id": self.layer_id,
            "support_size": self.support_size,
            "new_shape": self.new_shape
            })
        return config

class DecTransLayer(tf.keras.layers.Layer):
    """
    Tensorflow layer that reshapes the output of the previous layer into the required format for
    the next decoder. The dimension of the layers are 1) batch size, 2)repetitative part, 3) the
    input size of the decoder.
    """
    def __init__(self, layer_id, support_size, new_shape=None, **kwargs):
        super(DecTransLayer, self).__init__(**kwargs)
        self.layer_id = layer_id
        self.support_size = support_size
        self.new_shape = new_shape

    def build(self, input_shape):
        if self.layer_id % 2 == 0:
            hub = len(self.support_size) - int(self.layer_id / 2)
            if hub == len(self.support_size) - 1:
                condens = self.support_size[hub] + hub
            else:
                condens = self.support_size[hub]
            new_shape = (input_shape[0],
                         input_shape[1] * condens,
                         int(input_shape[2] / condens))

        if self.layer_id % 2 == 1:
            new_shape = (input_shape[0],
                         input_shape[1] * 2,
                         int(input_shape[2] /2))
        self.new_shape = new_shape
        # print(f"{self.name} input: {input_shape} output:{self.new_shape.numpy()}")

    def call(self, inputs):
        target_shape = (
            tf.shape(inputs)[0],
            self.new_shape[1],
            self.new_shape[2])
        return tf.reshape(inputs, target_shape)

    def get_config(self):
        config = super(DecTransLayer, self).get_config()
        config.update({
            "layer_id": self.layer_id,
            "support_size": self.support_size,
            "new_shape": self.new_shape
            })
        return config
    
class Hub0_encoder(tf.keras.layers.Layer):
    """
    Combines the features of the target node with the embedding of the target node and calculates
    a new embedding containing both the own features and neighbourhood.
    """

    def __init__(self, dim, act, seed=0, **kwargs):
        super(Hub0_encoder, self).__init__(**kwargs)
        self.dim = dim
        self.act = act
        self.seed = seed
        self.dense_layer = Dense(dim, activation=act, kernel_initializer=GlorotUniform(seed=seed))

    def call(self, inputs, training=False):
        """
        Args:
            inputs: tuple(neighbourhoods, features)
        """
        features = tf.expand_dims(inputs[1], -2)
        combined = tf.concat([features, inputs[0]], -1)
        return (self.dense_layer(combined, training=training), inputs[1])

    def get_config(self):
        config = super(Hub0_encoder, self).get_config()
        config.update({
            "dim": self.dim, 
            "act": self.act, 
            "seed": self.seed
            })
        return config

class Hub0_decoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, act, node_dims, seed=0, **kwargs):
        super(Hub0_decoder, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.act = act
        self.node_dims = node_dims
        self.seed = seed
        self.dense_layer = Dense(embedding_dim, activation=act, kernel_initializer=GlorotUniform(seed=seed))

    def call(self, inputs, training=False):
        """
        Args:
            inputs: embedding of the target node that contains information from own features and 
                    the neighbourhood.

        Returns: tuple with the reconstructed features and embedding of the neighbourhood.
        """
        combined = self.dense_layer(inputs[0], training=training)
        feat_out = combined[...,:self.node_dims]
        trans_layer = combined[...,self.node_dims:]
        return (trans_layer, feat_out)

    def get_config(self):
        config = super(Hub0_decoder, self).get_config()
        config.update({
            "embedding_dim": self.embedding_dim, 
            "act": self.act, 
            "node_dims": self.node_dims,
            "seed": self.seed
            })
        return config