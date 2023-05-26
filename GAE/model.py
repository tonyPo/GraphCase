#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:12:51 2019

@author: tonpoppe
"""

import os
import tensorflow as tf
import numpy as np
from GAE.transformation_layer import DecTransLayer, EncTransLayer, Hub0_encoder, Hub0Decoder
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Lambda
from tensorflow.keras.initializers import GlorotUniform

class GraphAutoEncoderModel(tf.keras.Model):
    """
    Directed graph implementation of GraphCase

    Args:
        learning_rate:  Learning rate used by the model.
        dims:       list with the dimension to be used for the layers.
        support_size: list with the sample size per layer.

    """

    def __init__(self, dims, support_size, feature_dim, verbose=False, seed=1, dropout=False,
        hub0_feature_with_neighb_dim=None, number_of_node_labels=0, act=tf.nn.sigmoid,
        useBN=False, encoder=None, decoder=None, hub0_encoder=None, hub0_decoder=None ):
 
        super(GraphAutoEncoderModel, self).__init__()
        self.dims = dims
        self.support_size = support_size
        self.verbose = verbose
        self.seed = seed
        self.act = act
        self.dropout = dropout
        self.useBN = useBN
        self.feature_dim = feature_dim
        self.number_of_node_labels = number_of_node_labels
        self.hub0_feature_with_neighb_dim = hub0_feature_with_neighb_dim
        self.sub_model_layer = None
        self.supervised_submodel = None
        self.mode = 'auto_encoder'

        if encoder is None:
            self.encoder = self.create_encoder()
            self.decoder = self.create_decoder()
            self.hub0_encoder = self.create_hub0_encoder()
            self.hub0_decoder = self.create_hub0_decoder()
        else:
            self.encoder = tf.keras.Model.from_config(encoder)
            self.decoder = tf.keras.Model.from_config(decoder)
            self.hub0_encoder = tf.keras.Model.from_config(hub0_encoder)
            self.hub0_decoder = tf.keras.Model.from_config(hub0_decoder)

    # def build(self, input_shape):
    #     """build model
    #     """
    #     self.encoder.build(input_shape)
    #     self.decoder.build(input_shape)

    def create_hub0_encoder(self): 
        if self.hub0_feature_with_neighb_dim is None:
            # return identity layer when no dimension is set
            return Lambda(lambda x:x)
        return Hub0_encoder(
            self.hub0_feature_with_neighb_dim, self.act, self.seed)

    def create_hub0_decoder(self):
        """Creates the decoder for the last layer. This laer take the embedding
        as input and return the feature values of the root node Xn
        and the input for the next layer to decoder Z

        Returns:
            _type_: Custom tensor keras layer
        """
        # return identity layer when no dimension is set
        if self.hub0_feature_with_neighb_dim is None:
            return Lambda(lambda x:x)
        dense_size = EncTransLayer.get_output_dim(
            len(self.dims)+1, self.support_size, self.dims, self.feature_dim)
        dense_size = dense_size + self.number_of_node_labels
        return Hub0Decoder(
                dense_size, self.act, self.number_of_node_labels, self.seed)

    def get_config(self):
        # config = super(GraphAutoEncoderModel, self).get_config()
        config = {
            "dims": self.dims,
            "support_size": self.support_size,
            "verbose": self.verbose,
            "seed": self.seed,
            "act": self.act,
            "dropout": self.dropout,
            "useBN": self.useBN,
            "feature_dim": self.feature_dim,
            "hub0_feature_with_neighb_dim": self.hub0_feature_with_neighb_dim,
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
            "hub0_encoder": self.hub0_encoder.get_config(),
            "hub0_decoder": self.hub0_decoder.get_config(),
            }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    
    def create_encoder(self):
        """
        create the encoder part of the model
        """
        encoder = tf.keras.models.Sequential()
        for i, d in enumerate(self.dims):
            if self.useBN:
                encoder.add(BatchNormalization())
            if self.dropout:
                encoder.add(Dropout(self.dropout))
            encoder.add(Dense(
                d,
                activation=self.act, 
                kernel_initializer=GlorotUniform(seed=self.seed)
            ))
            encoder.add(EncTransLayer(i+2, self.support_size, name="enctrans"+str(i)))
        return encoder

    def create_decoder(self):
        """
        creates the decoder part of the model
        """
        decoder = tf.keras.models.Sequential()
        for i in range(len(self.dims)-1, -1, -1):
            decoder.add(DecTransLayer(i+2, self.support_size, name="decTrans"+str(i)))
            if self.useBN:
                decoder.add(BatchNormalization())
            if self.dropout:
                decoder.add(Dropout(self.dropout))
            dense_size = EncTransLayer.get_output_dim(
                i+1, self.support_size, self.dims, self.feature_dim)
            decoder.add(Dense(
                dense_size, 
                activation=self.act, 
                kernel_initializer=GlorotUniform(seed=self.seed)
            ))
        return decoder

    def calculate_embedding(self, batch):
        """
        Calculates the embedding for the nodes in the specified bach.

        Args:
            batch: A tuple with node_ids, the features and neighbourhood.

        Returns: a 2d numpy matrix with in the first column the node ids following by the incoming
                embedding and then by the outgoing embedding.

        """
        x = self.encoder(batch[2])
        x, feat = self.hub0_encoder((x, batch[1]))
        node_id = tf.reshape(batch[0], (tf.shape(batch[0])[0], 1))
        embedding = np.hstack([node_id, tf.squeeze(x)])
        return embedding

    def call(self, inputs, training=False):
        """ apply model
        """
        if self.mode == 'auto_encoder':
            feature_hat = inputs[0]
            if self.sub_model_layer is not None:
                return (feature_hat, self.call_sub_model(self.sub_model_layer, inputs[1], training=True))
            x = self.encoder(inputs[1], training)
            x = self.hub0_encoder((x, inputs[0]), training)
            x, feature_hat = self.hub0_decoder(x, training)
            x = self.decoder(x, training)
            return (feature_hat, x)

        if self.mode == 'super':
            feature_hat = inputs[0]
            x = self.encoder(inputs[1], training)
            x = self.hub0_encoder((x, inputs[0]), training)
            x = self.supervised_submodel(x[0], training)
            return x

    def call_sub_model(self, layer_id, x, training=False):
        """ apply encoder without target node attributes
        """
        # enc_mdl = tf.keras.Model(
        #     inputs=self.encoder.input,
        #     outputs=self.encoder.get_layer("enctrans"+str(layer_id)).output)

        enc_mdl = tf.keras.models.Sequential()
        end_id = self.encoder.layers.index(
            self.encoder.get_layer("enctrans"+str(layer_id)))
        for i in range(0, end_id+1):
            enc_mdl.add(self.encoder.layers[i])

        dec_mdl = tf.keras.models.Sequential()
        start_id = self.decoder.layers.index(
            self.decoder.get_layer("decTrans"+str(layer_id)))
        for i in range(start_id, len(self.decoder.layers)):
            dec_mdl.add(self.decoder.layers[i])

        return dec_mdl(enc_mdl(x, training))
  

    def decode(self, embed):
        """
        Decodes the given embedding into a node and local neighbourhood.
        Args:
            embedding   : Embedding of the node

        Returns:
            A tuple with the node labels, inputlayer
        """   
        # reshape the input into 3-d, by repeating the row
        df_out = tf.reshape(np.float32(embed), [1, 1] +[len(embed)])
        x, feat = self.hub0_decoder((df_out, df_out))

        # check if feature are included in the embedding
        if self.hub0_feature_with_neighb_dim is None:
            feat = None
        x = self.decoder(x)

        return (feat, x)

    def create_supervised_model(self, supervised_submodel):
        self.supervised_submodel = supervised_submodel
        self.mode='super'
