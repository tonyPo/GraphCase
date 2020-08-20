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

# MAC OS bug
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class GraphAutoEncoderModel:
    """
    Directed graph implementation of GraphCase

    Args:
        learning_rate:  Learning rate used by the model.
        dims:       list with the dimension to be used for the layers.

    """

    def __init__(self,
                 learning_rate,
                 dims,
                 verbose=False,
                 seed=1):
        '''
            - feature_size: number of features
        '''
        self.dims = dims
        self.act = None
        self.verbose = verbose
        self.seed = seed
        self.layer_enc = {}
        self.layer_dec = {}

        self.features = None
        self.in_sample = None
        self.out_sample = None
        self.in_sample_amnt = None
        self.out_sample_amnt = None

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def save_layer(self, layer, filename):
        """
        Saves the specified layer into the specified filename

        Args:
            layer: number of the layer to be saved.
            filename: filename of the saved layer.
        """
        enc_layer = getattr(self, "layer" + str(layer) + "_enc")
        dec_layer = getattr(self, "layer" + str(layer) + "_dec")
        layers = (enc_layer, dec_layer)
        pickle.dump(layers, open(filename, "wb"))


    def load_layer(self, layer, filename):
        """
        Loads a layer from file.

        Args:
            layer: The layer number into which the saved layer is loaded.
            filename: The filename containing the saved layer.
        """
        layers = pickle.load(open(filename, "rb"))
        setattr(self, "layer" + str(layer) + "_enc", layers[0])
        setattr(self, "layer" + str(layer) + "_dec", layers[1])

    def __get_input_layer(self, batch):
        """
        Creates the input layer by applying a canoninal ordering of the local
        neighbourhood and deterministic sampling.

        Args:
            batch:  A tensor containing the nodes for which the input layer
                    needs to be constructed.
        """
        in_nodes, in_edges, in_weight = self.__blowup(batch, self.in_sample,
                                                      self.in_sample_amnt)
        out_nodes, out_edges, out_weight = self.__blowup(batch,
                                                         self.out_sample,
                                                         self.out_sample_amnt)

        node_ids = tf.concat([in_nodes, out_nodes], 1)
        node_labels = tf.nn.embedding_lookup(self.features, node_ids)

        edges = tf.concat([in_edges, out_edges], 1)
        nw_shape = (tf.shape(edges)[0].numpy(), tf.shape(edges)[1].numpy(), 1)

        weight = tf.concat([in_weight, out_weight], 1)

        # level 1 and level 2 combined in one list per level 1 node
        input_layer = tf.concat([tf.reshape(edges, nw_shape), node_labels], 2)
        input_layer = tf.cast(input_layer, tf.float32)
        return input_layer, weight

    def __blowup(self, batch, edge_lookup, weight_lookup):
        """
        Create a canonical ordering and sampling of the neighbourhood for the
        specified node id given a edge lookup table and edge weight lookup
        table.

        Args:
            batch:  A tensor containing the node ids for which the local
                    neighbourhood needs to be constructed.
            edge_lookup: A edge lookup tensor
            weight_lookup: An edge weight lookup tensor containing the node
                    id's of the sampled neighbourhood.

        Returns:
            A tuple with
            1) a tensor containing the node id's of the local
            neighbourhood of the nodes in the batch.
            2) A tensor containing the edge weight to be used as input
            3) A tensor containging the loss weight which is the product of
            edge weight on layer 1 and layer 2
        """
        # get nodes on level 1
        l1_node = tf.nn.embedding_lookup(edge_lookup, batch)
        l1_edge = tf.nn.embedding_lookup(weight_lookup, batch)

        # get nodes and weights of the incoming layer 2
        l2_in = tf.nn.embedding_lookup(self.in_sample, l1_node)
        l2_in_edge = tf.nn.embedding_lookup(self.in_sample_amnt, l1_node)

        # get nodes and weights of the outgoing layer 2
        l2_out = tf.nn.embedding_lookup(self.out_sample, l1_node)
        l2_out_edge = tf.nn.embedding_lookup(self.out_sample_amnt, l1_node)

        # add layer to layer 1 variables
        nw_shape = (tf.shape(l1_node)[0].numpy(), tf.shape(l1_node)[1].numpy(), 1)
        l1_node = tf.reshape(l1_node, nw_shape)
        l1_edge = tf.reshape(l1_edge, nw_shape)

        # get loss weights
        w1_w1 = tf.math.multiply(l1_edge, l1_edge)
        w2_in = l2_in_edge * l1_edge
        w2_out = l2_out_edge * l1_edge

        # level 1 and level 2 combined in one list per level 1 node
        combined = tf.concat([l1_node, l2_in, l1_node, l2_out], 2)
        # level 1 and level 2 combined in one list per level 1 node
        combined_edge = tf.concat([l1_edge, l2_in_edge, l1_edge, l2_out_edge], 2)
        combined_weights = tf.concat([w1_w1, w2_in, w1_w1, w2_out], 2)
        nw_shape = (tf.shape(combined)[0].numpy(),
                    tf.shape(combined)[1].numpy() *
                    tf.shape(combined)[2].numpy())
        flatten = tf.reshape(combined, nw_shape)
        flatten_edge = tf.reshape(combined_edge, nw_shape)
        flatten_weight = tf.reshape(combined_weights, nw_shape)

        return flatten, flatten_edge, flatten_weight

    def __set_up_layer(self, layer, input_layer):
        init_enc = tf.keras.initializers.GlorotUniform(seed=self.seed)
        init_dec = tf.keras.initializers.GlorotUniform(seed=self.seed)
        self.layer_enc[layer] = tf.keras.layers.Dense(self.dims[layer-1], activation=self.act,
                                                      use_bias=True, kernel_initializer=init_enc)
        self.layer_dec[layer] = tf.keras.layers.Dense(tf.shape(input_layer)[2],activation=self.act,
                                                      use_bias=True, kernel_initializer=init_dec)
        if self.verbose:
            print(f"Create layer {layer} output dim {self.dims[layer-1]}, ",
                  f"input dim {tf.shape(input_layer)[2]}")

    def __transform_input_layer(self, layer_id, previous_output):
        """
        Reshapes the output of the previous layer into the required format for the next
        encoder.

        Args:
            layer_id:   Layer number for which the input needs to be created.
            previous_output: Tensor containing the output of the previous encoder layer.

        Returns:
            reshaped tensor into the required format for the specified encoder layer.
        """
        n_size = tf.shape(self.in_sample)[1]
        if layer_id == 2:
            new_shape = (tf.shape(previous_output)[0],
                         2 * 2 * n_size,
                         (n_size+1) * tf.shape(previous_output)[2])

        if layer_id == 3:
            new_shape = (tf.shape(previous_output)[0],
                         2 * n_size,
                         2 * tf.shape(previous_output)[2])

        if layer_id == 4:
            new_shape = (tf.shape(previous_output)[0],
                         2,
                         n_size * tf.shape(previous_output)[2])

        return tf.reshape(previous_output, new_shape)

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

    def set_hyperparam(self, layer, dim=None, act=None, learning_rate=None):
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
        enc_out = {}
        enc_in = {}
        dec_out = {}
        dec_in = {}

         #create input layer
        enc_in[1], weight = self.__get_input_layer(batch)
        if (layer > 1) & (self.layer_enc.get(layer-1) is None):
            print(f"Please train layer {layer - 1} first")
            return

        with tf.GradientTape() as tape:
            for i in range(1, layer):
                enc_out[i] = self.layer_enc[i](enc_in[i])
                enc_in[i+1] = self.__transform_input_layer(i+1, enc_out[i])

            if self.layer_enc.get(layer) is None:
                self.__set_up_layer(layer, enc_in[layer])

            enc_out[layer] = self.layer_enc[layer](enc_in[layer])
            dec_out[layer] = self.layer_dec[layer](enc_out[layer])

            for i in range(layer-1, 0, -1):
                dec_in[i] = tf.reshape(dec_out[i+1], tf.shape(enc_out[i]))
                dec_out[i] = self.layer_dec[i](dec_in[i])

            loss = tf.keras.losses.MSE(enc_in[1], dec_out[1])
            loss = loss * tf.add(tf.dtypes.cast(weight, tf.float32), tf.constant(1.0))

            trainable_vars = None
            start = 1 if all_layers else layer
            for i in range(start, layer+1):
                if trainable_vars is None:
                    trainable_vars = self.layer_enc[i].trainable_variables
                else:
                    trainable_vars = trainable_vars + self.layer_enc[i].trainable_variables
                trainable_vars = trainable_vars + self.layer_dec[i].trainable_variables

            if not is_validation_step:
                grads = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return np.sum(loss), dec_out[1]

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
        if self.layer_enc.get(4) is None:
            print("Please train layer 4 first")
            return

        enc_in[1], _ = self.__get_input_layer(batch)
        enc_out[1] = self.layer_enc[1](enc_in[1])
        for i in range(2, len(self.dims)+1):
            enc_in[i] = self.__transform_input_layer(i, enc_out[i-1])
            enc_out[i] = self.layer_enc[i](enc_in[i])

        node_id = tf.reshape(batch, (tf.shape(batch)[0], 1))
        embedding = np.hstack([node_id, self.get_embedding(enc_out[4])])
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
        self.layer_enc[layer] = None
        self.layer_dec[layer] = None
