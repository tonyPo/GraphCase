#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:12:51 2019

@author: tonpoppe
"""

import os
import tensorflow as tf
import numpy as np
from GAE.transformation_layer import DecTransLayer, EncTransLayer, Hub0_encoder, Hub0_decoder
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.initializers import GlorotUniform

# MAC OS bug
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
        useBN=False, encoder=None, decoder=None, hub0_encoder=None, hub0_decoder=None):
        '''
            - feature_size: number of features
        '''
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

        self.features = None
        self.in_sample = None
        self.out_sample = None
        self.in_sample_amnt = None
        self.out_sample_amnt = None

        self.encoder = self.create_encoder() if encoder is None else encoder
        self.decoder = self.create_decoder() if decoder is None else decoder
        self.hub0_encoder = hub0_encoder
        self.hub0_decoder = hub0_decoder
        
        if hub0_feature_with_neighb_dim is not None and hub0_encoder is None:
            self.hub0_encoder = Hub0_encoder(       
                hub0_feature_with_neighb_dim, act, seed
            )
            dense_size = EncTransLayer.get_output_dim(
                len(dims)+1, support_size, dims, feature_dim)
            dense_size = dense_size + number_of_node_labels
            self.hub0_decoder = Hub0_decoder(
                dense_size, act, feature_dim, seed
            )

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
            "encoder": self.encoder,
            "decoder": self.decoder,
            "hub0_encoder": self.hub0_encoder,
            "hub0_decoder": self.hub0_decoder
            }
        return config
    
    def create_encoder(self):
        """
        create the encoder part of the model
        """
        encoder = tf.keras.models.Sequential()
        for i, d in enumerate(self.dims):
            if self.useBN:
                encoder.add(BatchNormalization)
            if self.dropout:
                encoder.add(Dropout)
            encoder.add(Dense(
                d,
                activation=self.act, 
                kernel_initializer=GlorotUniform(seed=self.seed)
            ))
            encoder.add(EncTransLayer(i+2, self.support_size, name="enctrans"+str(i+2)))
        return encoder

    def create_decoder(self):
        """
        creates the decoder part of the model
        """
        decoder = tf.keras.models.Sequential()
        for i in range(len(self.dims)-1, -1, -1):
            decoder.add(DecTransLayer(i+2, self.support_size, name="decTrans"+str(i+3)))
            if self.useBN:
                decoder.add(BatchNormalization)
            if self.dropout:
                decoder.add(Dropout)
            dense_size = EncTransLayer.get_output_dim(
                i+1, self.support_size, self.dims, self.feature_dim)
            decoder.add(Dense(
                dense_size, 
                activation=self.act, 
                kernel_initializer=GlorotUniform(seed=self.seed)
            ))
        return decoder

    def __get_next_hub(self, node_ids, hub, direction, feat, weight):
        """
        Retrieves the features including edge weights for the specified hub and direction.
        The features are combined with the features of the lower hubs.

        Args:
            node_ids:   A tensor containing the node_ids for which the featurs need to be
                        retrieved.
            hub:        integer specifying the hub for which the features need to be retrieved.
            direction:  {in, out} indicator whether the features for the incoming or outgoing
                        neighbourhood need to be retrieved.
            feat:       A tensor with the features of the lower
                        hub which will be combined in this hub.
            weight:     A tensor with the weights of the lower hubs
                        which will be used for multiplication of this hubs weights.

        Returns:
            A tuple of two tensors with 1) a tensor with the features and 2 a tensor with the
            corresponding weights.

        """
        support = self.support_size[hub-1]
        if direction == 'in':
            sample_node = self.in_sample[:, :support]
            sample_weight = self.in_sample_amnt[:, :support]
        else:
            sample_node = self.out_sample[:, :support]
            sample_weight = self.out_sample_amnt[:, :support]

        next_nodes = tf.nn.embedding_lookup(sample_node, node_ids)
        weight_next = tf.nn.embedding_lookup(sample_weight, node_ids)
        feat_next = tf.nn.embedding_lookup(self.features, next_nodes)
        #combine feature + edge labels
        feat_next = tf.concat([weight_next, feat_next], -1)
        weight_next = tf.slice(weight_next, tf.repeat([0], tf.shape(tf.shape(weight_next))),
                               tf.concat([tf.shape(weight_next)[:-1], [1]], axis=0))
        weight_next = tf.squeeze(weight_next)
        # check weight_next = weight_next[...,0]

        if hub < len(self.support_size):
            # add additional dimension for the next hub features and weight
            feat_next = tf.expand_dims(feat_next, -2)
            weight_next = tf.expand_dims(weight_next, -1)
            if feat is not None:
                shape = tf.shape(feat)
                # tile_shape = [1] * (len(shape) -2) + [self.support_size[hub - 1]] + [1]
                tile_shape = tf.concat([
                    tf.repeat([1], tf.shape(shape)-2),
                    [self.support_size[hub - 1]],
                    [1]
                    ], axis=0
                )
                feat = tf.tile(feat, tile_shape)
                feat = tf.expand_dims(feat, -2)
                feat_next = tf.concat([feat, feat_next], -2)
                weight = tf.tile(weight, tile_shape[:-1])
                weight = tf.expand_dims(weight, -1)
                weight_next = tf.concat([weight, weight_next], -1)


            feat_next, weight_next = self.get_input_layer(next_nodes, hub+1,
                                                            feat_next, weight_next)

        return feat_next, weight_next

    def get_input_layer(self, node_ids, hub, feat=None, weight=None):
        """
        Retrieve the first input layer by sampling the graph. This method is called
        recursively per hub.

        Args:
            node_ids:   tensor with the node ids for which the input_layer needs to be calculated.
            hub:        hub of the inputlayer, for hub = 1.
            feat:       In case this is called recursively, a tensor with the features of the lower
                        hub which will be combined in this hub.
            weight:     In case of a recursive call, a tensor with the weights of the lower hubs
                        which will be used for multiplication of this hubs weights.

        returns:
            a tuple with 1) a tensor of the features for the specified hub and 2) a tensor with
            the weights for the specified hub.
        """
        next_in_feat, next_in_weight = self.__get_next_hub(node_ids, hub, 'in', feat, weight)
        next_out_feat, next_out_weight = self.__get_next_hub(node_ids, hub, 'out', feat, weight)

        if feat is not None:
            if hub == len(self.support_size):
                feat = tf.concat([feat, next_in_feat, feat, next_out_feat], -2)

                shape = tf.shape(weight)
                head = None
                # for level in range(shape[-1]):
                for level in range(hub - 1):
                    factor = tf.slice(weight, 
                        tf.concat([tf.repeat([0], tf.shape(shape)-1), [level]], axis=0),
                        tf.concat([shape[:-1], [1]], axis=0))
                    next_in_weight = tf.math.multiply(next_in_weight, factor)
                    next_out_weight = tf.math.multiply(next_out_weight, factor)
                    if head is None:
                        head = tf.math.pow(factor, tf.cast(shape[-1] - level + 1, tf.float32))
                        tail = tf.slice(weight,
                            tf.concat([tf.repeat([0], tf.shape(shape)-1), [level+1]], axis=0),
                            tf.concat([shape[:-1], [shape[-1] - 1]], axis=0))
                        tail = tf.math.multiply(tail, factor)
                    else:
                        tail = tf.math.multiply(tail, factor)
                        head_add = tf.slice(tail,
                            tf.concat([tf.repeat([0], tf.shape(shape)-1), [0]], axis=0),
                            tf.concat([shape[:-1], [1]], axis=0))
                        head = tf.concat([head, head_add], -1)
                        tail = tf.slice(tail,
                            tf.concat([tf.repeat([0], tf.shape(shape)-1), [1]], axis=0),
                            tf.concat([shape[:-1], [shape[-1] - 1 - level]], axis=0))

                weight_comb = tf.concat([head, next_in_weight, head, next_out_weight], -1)

            else:
                feat = tf.concat([next_in_feat, next_out_feat], -2)
                weight_comb = tf.concat([next_in_weight, next_out_weight], -1)

            shape = tf.shape(feat)
            new_shape = tf.concat([shape[:-3], [shape[-3] * shape[-2], shape[-1]]], axis=0)
            feat = tf.reshape(feat, new_shape)
            weight = tf.reshape(weight_comb, new_shape[:-1])

        else:
            feat = tf.concat([next_in_feat, next_out_feat], -2)
            weight = tf.concat([next_in_weight, next_out_weight], -1)

        return feat, weight

    def calculate_embedding(self, batch):
        """
        Calculates the embedding for the nodes in the specified bach.

        Args:
            batch: A tensor containing the node ids for which the embedding needs to calculated.

        Returns: a 2d numpy matrix with in the first column the node ids following by the incoming
                embedding and then by the outgoing embedding.

        """
        df_out, _ = self.get_input_layer(batch, hub=1)
        x = self.encoder(df_out)

        node_id = tf.reshape(batch, (tf.shape(batch)[0], 1))
        embedding = np.hstack([node_id, tf.squeeze(x)])
        return embedding

    def set_constant_data(self, features, in_sample, out_sample, in_sample_weight,
                          out_sample_weight):
        """
        Set the constant input data of the model.

        Args:
            features:   a 2d numpy dataframe containing the feature information.
            in_sample:  a 2d numpy matrix with the node ids of the incoming neighbourhood.
            out_sample: a 2d numpy matrix with the node ids of the outgoing neighbourhood.
            in_sample_weight: a 3d numpy matrix with the sample weight of the incoming
                        neighbourhood. (batch_size, support_size, len(edge_labels))
            out_sample_weight: a 2d numpy matrix with the sample weight of the outgoing
                        neighbourhood. (batch_size, support_size, len(edge_labels))
        """

        self.features = tf.constant(features, name="features")
        self.in_sample = tf.constant(in_sample, dtype=tf.int64, name="in_sample")
        self.out_sample = tf.constant(out_sample, dtype=tf.int64, name="out_sample")
        self.in_sample_amnt = tf.constant(in_sample_weight, name="in_sample_amnt")
        self.out_sample_amnt = tf.constant(out_sample_weight, name="out_sample_amnt")

    # def __is_combination_layer(self, layer):
    #     """
    #     checks if the layer is a layer in which the target is combined with the embeding of the
    #     in and outgoing neighbourhood. The combination layer is optional and can be identified
    #     by the lenght of the dims, i.e. the dimension list has one additional dimension specifying
    #     the embedding size of the combination.
    #     """
    #     return (len(self.dims) - 1 == len(self.support_size) * 2) and (layer == len(self.dims))

    def __add_hub0_features(self, input_layer, batch):
        """
        Adjusts the input layer for the final combination layer in which the features of the target
        node are combined with the in and outgoing embedding by adding the features of the target
        to the input layer.
        """
        features = tf.nn.embedding_lookup(self.features, batch)
        features = tf.expand_dims(features, -2)
        trans_layer = tf.concat([features, input_layer], -1)

        return features, trans_layer

    def __extract_hub0_features(self, out_layer):
        """
        Adjust the output_layer of the final combination layer which combines the features with the
        in and outgoing neighbourhood. The layer is adjusted by removing the values related to the
        target features.
        """

        feature_size = tf.shape(self.features)[1]
        out_shape = tf.shape(out_layer)
        # feat_out = tf.slice(out_layer, [0, 0, 0], tf.concat([out_shape[:-1] + [feature_size]], axis=0)
        # trans_layer = tf.slice(out_layer, [0, 0, feature_size], out_shape[:-1] + [-1])
        feat_out = out_layer[...,:feature_size]
        trans_layer = out_layer[...,feature_size:]
        return feat_out, trans_layer

    def get_features(self, batch):
        return tf.nn.embedding_lookup(self.features, batch)

    def call(self, inputs):
        # x, _ = self.get_input_layer(inputs, hub=1)
        # x = tf.keras.layers.Lambda(lambda x: self.get_input_layer(x, hub=1))(inputs)
        x = self.encoder(inputs[1])
        if self.hub0_feature_with_neighb_dim is not None:
            x = self.hub0_encoder((x, inputs[0]))
            feature_hat, x = self.hub0_decoder(x)
        x = self.decoder(x)

        return (feature_hat, x)

    def train_step_old(self, data):
        # custom training step
        features, neighbourhood = data

        with tf.GradientTape() as tape:
            x = self.encoder(neighbourhood, training=True)
            if self.hub0_feature_with_neighb_dim is not None:
                features, x = self.__add_hub0_features(x, n)
                x = self.hub0_encoder(x, training=True)
                x = self.hub0_decoder(x, training=True)
                feat_pred, x = self.__extract_hub0_features(x)
                self.compiled_loss(
                    features,
                    feat_pred
                )
            y_pred = self.decoder(x, training=True)

            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=w,
                regularization_losses=self.losses,
            )

                # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=w)

        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        n, y = data
        # Compute predictions
        x, w = self.get_input_layer(n, hub=1)
        x = self.encoder(x, training=False)
        if self.hub0_feature_with_neighb_dim is not None:
            features, x = self.__add_hub0_features(x, n)
            x = self.hub0_encoder(x, training=False)
            x = self.hub0_decoder(x, training=False)
            feat_pred, x = self.__extract_hub0_features(x)
            self.compiled_loss(
                    features,
                    feat_pred
                )
        y_pred = self.decoder(x, training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, sample_weight=w, regularization_losses=self.losses)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def decode(self, embed):
        """
        Decodes the given embedding into a node and local neighbourhood.
        Args:
            embedding   : Embedding of the node

        Returns:
            A tuple with the node labels, inputlayer
        """
        if self.layer_enc.get(str(len(self.dims))) is None:
            print("Please train neural net first")
            return
        
        # reshape the input into 3-d, by repeating the row
        df_out = tf.reshape(np.float32(embed), [1, 1] +[len(embed)])
        df_out = tf.tile(df_out, [self.trans_dec[str(len(self.dims))].nw_shape[0]] + [1, 1])

        layers = len(self.dims)
        feat_out = None
        for layer in range(layers, 0, -1):
            if self.useBN:
                df_out = self.BN_dec[str(layer)](df_out, training=False)
            # if self.dropout is not None:
            #     df_out = self.dropout(df_out, training=False)
            df_out = self.layer_dec[str(layer)](df_out)

            if self.__is_combination_layer(layer):
                feat_out, df_out = self.__extract_hub0_features(df_out)
            if layer > 1:
                df_out = self.trans_dec[str(layer)](df_out)

        return tf.squeeze(feat_out[0]), df_out[0:1, :, :]