#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:12:51 2019

@author: tonpoppe
"""

import os
import pickle
import tensorflow as tf
import numpy as np
from GAE.transformation_layer import DecTransLayer, EncTransLayer

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

    def __init__(self, learning_rate, dims, support_size, verbose=False, seed=1, dropout=None,
                 act=tf.nn.sigmoid):
        '''
            - feature_size: number of features
        '''
        super(GraphAutoEncoderModel, self).__init__()
        self.dims = dims
        self.support_size = support_size
        self.act = None
        self.verbose = verbose
        self.seed = seed
        self.act = act
        self.layer_enc = {}
        self.layer_dec = {}
        self.trans_enc = {}
        self.trans_dec = {}

        self.features = None
        self.in_sample = None
        self.out_sample = None
        self.in_sample_amnt = None
        self.out_sample_amnt = None

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)
        else:
            self.dropout = None


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
        feat_next = tf.concat([weight_next, feat_next], -1)
        weight_next = tf.slice(weight_next, [0] * len(tf.shape(weight_next)),
                               tf.shape(weight_next).numpy().tolist()[:-1] + [1])
        weight_next = tf.squeeze(weight_next)

        if hub < len(self.support_size):
            feat_next = tf.expand_dims(feat_next, -2)
            weight_next = tf.expand_dims(weight_next, -1)
            if feat is not None:
                shape = tf.shape(feat).numpy().tolist()
                tile_shape = [1] * (len(shape) -2) + [self.support_size[hub - 1]] + [1]
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

                shape = tf.shape(weight).numpy().tolist()
                head = None
                for level in range(shape[-1]):
                    factor = tf.slice(weight, [0] * (len(shape)-1) + [level], shape[:-1] + [1])
                    next_in_weight = tf.math.multiply(next_in_weight, factor)
                    next_out_weight = tf.math.multiply(next_out_weight, factor)
                    if head is None:
                        head = tf.math.pow(factor, shape[-1] - level + 1)
                        tail = tf.slice(weight,
                                        [0] * (len(shape)-1) + [level + 1],
                                        shape[:-1] + [shape[-1] - 1])
                        tail = tf.math.multiply(tail, factor)
                    else:
                        tail = tf.math.multiply(tail, factor)
                        head_add = tf.slice(tail, [0] * (len(shape)-1) + [0], shape[:-1] + [1])
                        head = tf.concat([head, head_add], -1)
                        tail = tf.slice(tail,
                                        [0] * (len(shape)-1) + [1],
                                        shape[:-1] + [shape[-1] -1 - level])

                weight_comb = tf.concat([head, next_in_weight, head, next_out_weight], -1)

            else:
                feat = tf.concat([next_in_feat, next_out_feat], -2)
                weight_comb = tf.concat([next_in_weight, next_out_weight], -1)

            shape = tf.shape(feat).numpy().tolist()
            new_shape = shape[:-3] + [shape[-3] * shape[-2], shape[-1]]
            feat = tf.reshape(feat, new_shape)
            weight = tf.reshape(weight_comb, new_shape[:-1])

        else:
            feat = tf.concat([next_in_feat, next_out_feat], -2)
            weight = tf.concat([next_in_weight, next_out_weight], -1)

        return feat, weight


    def __set_up_layer(self, layer, input_layer):
        init_enc = tf.keras.initializers.GlorotUniform(seed=self.seed)
        init_dec = tf.keras.initializers.GlorotUniform(seed=self.seed)
        enc_input_layer = input_layer.numpy().tolist()[:-1] + [self.dims[layer-1]]
        self.layer_enc[str(layer)] = tf.keras.layers.Dense(self.dims[layer-1], activation=self.act,
                                                      use_bias=True, input_shape=input_layer,
                                                      kernel_initializer=init_enc)
        self.layer_dec[str(layer)] = tf.keras.layers.Dense(input_layer[2],
                                                           input_shape=enc_input_layer,
                                                      activation=self.act,
                                                      use_bias=True, kernel_initializer=init_dec)
        if self.verbose:
            print(f"Create layer {layer} output dim {self.dims[layer-1]}, ",
                  f"input dim {input_layer[2]}")


    def get_embedding(self, layer4_enc_out):
        """
        Transforms the output of the last encoder to combine the incoming and outgoing
        layer into one embedding.

        Args:
            layer4_enc_out: a 3d tensor with dimension 1 the node_ids, dimension 2 the number of
                            neighbourhoods per node, i.e. incoming and outgoing and dimension 3
                            equal to the embedding per neighbourhood.
        """
        new_shape = (tf.shape(layer4_enc_out)[0],
                     tf.shape(layer4_enc_out)[1] * tf.shape(layer4_enc_out)[2])
        return tf.reshape(layer4_enc_out, new_shape)

    def set_hyperparam(self, layer, dim=None, act=None, learning_rate=None, dropout=None):
        """
        Sets the hyperparameters for the model. The dimension is only updated for the specified
        layer.

        Args:
            layer: integer indicating the layer for which the dimension needs updating.
            dim: integer indicating the size of the embedding for the specified layer.
            act: activation function to be used by the model.
            learning_rate: learning rate of the optimizer used for training the model.
        """
        if dim is not None:
            self.dims[layer-1] = dim
        if act is not None:
            if act == 'iden':
                self.act = None
            else:
                self.act = act
        if learning_rate is not None:
            self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(dropout)

    def train_layer(self, layer, batch, all_layers=False, is_validation_step=False):
        """
        Trains the specified layer of the model. If the all_layers indicator is set, then all lower
        layers are included in the training. When the is_validation indicator is set then the
        weights of the MLPs are not updated, but only the loss is calculated.

        Args:
            layer:  integer indicating the layer to be trained.
            batch:  tensor with the node_ids used for training.
            all_layers: boolean indicating if the weights of the lower layers are included in the
                    training.
            is_validation_step: boolean, when set to true only the loss is calculated without
                    updating the weights of the layers.
        """

         #create input layer
        df_in, weight = self.get_input_layer(batch, hub=1)

        if (layer > 1) & (self.layer_enc.get(str(layer-1)) is None):
            print(f"Please train layer {layer - 1} first")
            return

        with tf.GradientTape() as tape:
            df_out = df_in
            for i in range(1, layer):
                if self.dropout is not None:
                    df_out = self.dropout(df_out, training=not is_validation_step)
                df_out = self.layer_enc[str(i)](df_out)
                if self.trans_enc.get(str(i+1)) is None:
                    self.trans_enc[str(i+1)] = EncTransLayer(i+1, self.support_size, tf.shape(df_out))
                    self.trans_dec[str(i+1)] = DecTransLayer(tf.shape(df_out))

                df_out = self.trans_enc[str(i+1)](df_out)
        
            if self.__is_combination_layer(layer):
                feat_in, df_out = self.__add_hub0_features(df_out, batch)

            if self.layer_enc.get(str(layer)) is None:
                self.__set_up_layer(layer, tf.shape(df_out))

            if self.dropout is not None:
                df_out = self.dropout(df_out, training=not is_validation_step)
            df_out = self.layer_enc[str(layer)](df_out)
            if self.dropout is not None:
                df_out = self.dropout(df_out, training=not is_validation_step)
            df_out = self.layer_dec[str(layer)](df_out)

            if self.__is_combination_layer(layer):
                feat_out, df_out = self.__extract_hub0_features(df_out)

            for i in range(layer-1, 0, -1):
                df_out = self.trans_dec[str(i+1)](df_out)
                # tf.reshape(dec_out[i+1], tf.shape(enc_out[i]))
                if self.dropout is not None:
                    df_out = self.dropout(df_out, training=not is_validation_step)
                df_out = self.layer_dec[str(i)](df_out)

            loss = tf.keras.losses.MSE(df_in, df_out)
            loss = loss * tf.add(tf.dtypes.cast(weight, tf.float32), tf.constant(1.0))
            if self.__is_combination_layer(layer):
                loss_feat = tf.keras.losses.MSE(feat_in, feat_out)
                loss = tf.concat([loss, loss_feat], -1)

            trainable_vars = None
            start = 1 if all_layers else layer
            for i in range(start, layer+1):
                if trainable_vars is None:
                    trainable_vars = self.layer_enc[str(i)].trainable_variables
                else:
                    trainable_vars = trainable_vars + self.layer_enc[str(i)].trainable_variables
                trainable_vars = trainable_vars + self.layer_dec[str(i)].trainable_variables

            if not is_validation_step:
                grads = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return np.sum(loss), df_out

    def calculate_embedding(self, batch):
        """
        Calculates the embedding for the nodes in the specified bach.

        Args:
            batch: A tensor containing the node ids for which the embedding needs to calculated.

        Returns: a 2d numpy matrix with in the first column the node ids following by the incoming
                embedding and then by the outgoing embedding.

        """
        enc_in = {}
        enc_out = {}
        if self.layer_enc.get(str(len(self.dims))) is None:
            print("Please train layer 4 first")
            return

        df_out, _ = self.get_input_layer(batch, hub=1)
        # filename = '/Users/tonpoppe/workspace/GraphCase/data/enc1_gb.txt'
        # sh = tf.shape(enc_in[1]).numpy().tolist()
        # res = np.reshape(enc_in[1].numpy(), (sh[0], sh[1] * sh[2]))
        # np.savetxt(filename, res, delimiter='|')
        df_out = self.layer_enc[str(1)](df_out)
        for i in range(2, len(self.dims)+1):
            df_out = self.trans_enc[str(i)](df_out)
            if self.__is_combination_layer(i):
                _, df_out = self.__add_hub0_features(df_out, batch)
            df_out = self.layer_enc[str(i)](df_out)

        node_id = tf.reshape(batch, (tf.shape(batch)[0], 1))
        embedding = np.hstack([node_id, self.get_embedding(df_out)])
        return embedding

    def set_constant_data(self, features, in_sample, out_sample, in_sample_weight,
                          out_sample_weight):
        """
        Set the constant input data of the model.

        Args:
            features:   a 2d numpy dataframe containing the feature information.
            in_sample:  a 2d numpy matrix with the node ids of the incoming neighbourhood.
            out_sample: a 2d numpy matrix with the node ids of the outgoing neighbourhood.
            in_sample_weight: a 2d numpy matrix with the sample weight of the incoming
                        neighbourhood.
            out_sample_weight: a 2d numpy matrix with the sample weight of the outgoing
                        neighbourhood.

        """

        self.features = tf.constant(features, name="features")
        self.in_sample = tf.constant(in_sample, dtype=tf.int64, name="in_sample")
        self.out_sample = tf.constant(out_sample, dtype=tf.int64, name="out_sample")
        self.in_sample_amnt = tf.constant(in_sample_weight, name="in_sample_amnt")
        self.out_sample_amnt = tf.constant(out_sample_weight, name="out_sample_amnt")

    def reset_layer(self, layer):
        """
        Resets the specified layer.
        """
        self.layer_enc[str(layer)] = None
        self.layer_dec[str(layer)] = None

    def __is_combination_layer(self, layer):
        """
        checks if the layer is a layer in which the target is combined with the embeding of the
        in and outgoing neighbourhood. The combination layer is optional and can be identified
        by the lenght of the dims, i.e. the dimension list has one additional dimension specifying
        the embedding size of the combination.
        """
        return (len(self.dims) - 1 == len(self.support_size) * 2) and (layer == len(self.dims))

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
        out_shape = tf.shape(out_layer).numpy().tolist()
        feat_out = tf.slice(out_layer, [0, 0, 0], out_shape[:-1] + [feature_size])
        trans_layer = tf.slice(out_layer, [0, 0, feature_size], out_shape[:-1] + [-1])
        return feat_out, trans_layer

    def call(self, inputs):
        return self.get_embedding(inputs)
