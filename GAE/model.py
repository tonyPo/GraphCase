#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:12:51 2019

@author: tonpoppe
"""

# MAC OS bug
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys

# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import pickle



        
class GraphAutoEncoderModel:
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self,
                  learning_rate,
                 dims,
                  feature_size=None,
                  verbose=False,
                  **kwargs):
        
        # removed: features, adj, degrees, 
        '''
            - feature_size: number of features
        '''
        self.dims = dims
        self.act = None
        self.verbose=verbose
        self.layer1_enc = None
        self.layer2_enc = None
        self.layer3_enc = None
        self.layer4_enc = None
        self.layer1_dec = None
        self.layer2_dec = None
        self.layer3_dec = None
        self.layer4_dec = None

        # set dimensions of the hidden layers h(0) to h(k)
        self.samples = [] #variables for feature data in batch 1
        self.s_iter = None #batch iterators
        self.train_init_op = [] # list of training initialisators
        self.val_init_op = [] # list of validation initialisators
        self.inc_init_op = [] # list of incremental initialisators

        self.features = None
        self.in_sample = None
        self.out_sample = None
        self.in_sample_amnt = None
        self.out_sample_amnt = None

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def save_layer(self, layer, filename):
        enc_layer= getattr(self, "layer" + str(layer) + "_enc")
        dec_layer= getattr(self, "layer" + str(layer) + "_dec")
        layers = (enc_layer, dec_layer)
        pickle.dump(layers, open( filename, "wb"))


    def load_layer(self, layer, filename):
        layers = pickle.load(open( filename, "rb"))
        setattr(self, "layer" + str(layer) + "_enc", layers[0])
        setattr(self, "layer" + str(layer) + "_dec", layers[1])

    def get_input_layer(self, batch):
        in_nodes, in_edges, in_weight = self.blowup(batch, self.in_sample, self.in_sample_amnt )
        out_nodes, out_edges, out_weight = self.blowup(batch, self.out_sample,  self.out_sample_amnt)

        node_ids = tf.concat([in_nodes, out_nodes],1)
        node_labels = tf.nn.embedding_lookup(self.features, node_ids)

        edges = tf.concat([in_edges, out_edges],1)
        nw_shape = (tf.shape(edges)[0].numpy(), tf.shape(edges)[1].numpy(), 1)

        weight = tf.concat([in_weight, out_weight],1)

        input_layer = tf.concat([tf.reshape(edges,nw_shape), node_labels], 2) # level 1 and level 2 combined in one list per level 1 node
        # print(f"applied canonical ordering resulting in {tf.shape(input_layer)[0]} by  {tf.shape(input_layer)[1]} dataset")
        input_layer = tf.cast(input_layer, tf.float32) 
        return input_layer, weight

    def blowup(self, batch, lookup, edge_lookup):
        # get nodes on level 1
        l1 = tf.nn.embedding_lookup(lookup, batch)
        l1_edge = tf.nn.embedding_lookup(edge_lookup, batch)

        # get in layer 2
        l2_in = tf.nn.embedding_lookup(self.in_sample, l1)    # get nodes on level 2
        l2_in_edge = tf.nn.embedding_lookup(self.in_sample_amnt, l1)

        # get out layer 2
        l2_out = tf.nn.embedding_lookup(self.out_sample, l1)    # get nodes on level 2
        l2_out_edge = tf.nn.embedding_lookup(self.out_sample_amnt, l1)

        # add layer to layer 1 variables
        nw_shape = (tf.shape(l1)[0].numpy(), tf.shape(l1)[1].numpy(), 1)
        l1 = tf.reshape(l1, nw_shape)
        l1_edge = tf.reshape(l1_edge, nw_shape)

        # get loss weights
        w1 = tf.math.multiply(l1_edge, l1_edge)
        w2_in = l2_in_edge * l1_edge
        w2_out = l2_out_edge * l1_edge

        combined = tf.concat([l1, l2_in, l1, l2_out], 2)  # level 1 and level 2 combined in one list per level 1 node
        combined_edge = tf.concat([l1_edge, l2_in_edge, l1_edge, l2_out_edge], 2)  # level 1 and level 2 combined in one list per level 1 node
        combined_weights = tf.concat([w1, w2_in, w1, w2_out], 2)
        nw_shape = (tf.shape(combined)[0].numpy(), tf.shape(combined)[1].numpy() * tf.shape(combined)[2].numpy() )
        flatten = tf.reshape(combined, nw_shape)  # all nodes in one list
        flatten_edge = tf.reshape(combined_edge, nw_shape)  # all nodes in one list
        flatten_weight = tf.reshape(combined_weights, nw_shape)

        return flatten, flatten_edge, flatten_weight

    def set_up_layer1(self, input_layer):
        self.layer1_enc = tf.keras.layers.Dense(self.dims[0], activation=self.act, use_bias=True)
        self.layer1_dec = tf.keras.layers.Dense(tf.shape(input_layer)[2], activation=self.act, use_bias=True)
        if self.verbose:
            print(f"Create layer1 output dim {self.dims[0]}, input dim {tf.shape(input_layer)[2]}")

    def set_up_layer2(self, layer2_input):
        self.layer2_enc = tf.keras.layers.Dense(self.dims[1], activation=self.act, use_bias=True)
        self.layer2_dec = tf.keras.layers.Dense(tf.shape(layer2_input)[2], activation=self.act, use_bias=True)
        if self.verbose:
            print(f"Create layer2 output dim {self.dims[1]}, input dim{tf.shape(layer2_input)[2]}")

    def set_up_layer3(self, layer3_input):
        self.layer3_enc = tf.keras.layers.Dense(self.dims[2], activation=self.act, use_bias=True)
        self.layer3_dec = tf.keras.layers.Dense(tf.shape(layer3_input)[2], activation=self.act, use_bias=True)
        if self.verbose:
            print(f"Create layer3 output dim {self.dims[2]}, input dim{tf.shape(layer3_input)[2]}")

    def set_up_layer4(self, layer4_input):
        self.layer4_enc = tf.keras.layers.Dense(self.dims[3], activation=self.act, use_bias=True)
        self.layer4_dec = tf.keras.layers.Dense(tf.shape(layer4_input)[2], activation=self.act, use_bias=True)
        if self.verbose:
            print(f"Create layer4 output dim {self.dims[3]}, input dim{tf.shape(layer4_input)[2]}")

    def get_input_layer2(self, layer1_enc_out):
        n_size = tf.shape(self.in_sample)[1]
        new_shape = (tf.shape(layer1_enc_out)[0],
                     2*2 * n_size,
                     (n_size+1) * tf.shape(layer1_enc_out)[2])
        layer2_input = tf.reshape(layer1_enc_out, new_shape)
        return layer2_input

    def get_input_layer3(self, layer2_enc_out):
        n_size = tf.shape(self.in_sample)[1]
        new_shape = (tf.shape(layer2_enc_out)[0],
                     2*n_size,
                     2*tf.shape(layer2_enc_out)[2])
        layer3_input = tf.reshape(layer2_enc_out, new_shape)
        return layer3_input

    def get_input_layer4(self, layer3_enc_out):
        n_size = tf.shape(self.in_sample)[1]
        new_shape = (tf.shape(layer3_enc_out)[0],
                     2,
                     n_size*tf.shape(layer3_enc_out)[2])
        layer4_input = tf.reshape(layer3_enc_out, new_shape)
        return layer4_input

    def get_embedding(self, layer4_enc_out ):
        new_shape = (tf.shape(layer4_enc_out)[0],
                     tf.shape(layer4_enc_out)[1] * tf.shape(layer4_enc_out)[2])
        return tf.reshape(layer4_enc_out, new_shape)

    def set_hyperparam(self, layer, dim=None, act=None, learning_rate=None):
        if dim is not None:
            self.dims[layer-1] = dim
        if act is not None:
            if act == 'iden':
                self.act = None
            else:
                self.act = act
        if learning_rate is not None:
            self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    def train_layer1(self, batch, is_validation_step=False):
        #create input layer
        input_layer, weight = self.get_input_layer(batch)
        if self.layer1_enc is None:
            self.set_up_layer1(input_layer)
        with tf.GradientTape() as tape:
            layer1_enc_out = self.layer1_enc(input_layer)
            outputs1 = self.layer1_dec(layer1_enc_out)
            # loss = self._loss(input_layer, outputs1 )
            loss = tf.keras.losses.MSE( input_layer, outputs1)
            loss = loss * tf.add(tf.dtypes.cast(weight, tf.float32), tf.constant(1.0))
            if not is_validation_step:
                grads = tape.gradient(loss, self.layer1_enc.trainable_variables + self.layer1_dec.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.layer1_enc.trainable_variables +
                                                          self.layer1_dec.trainable_variables))
        return np.sum(loss), outputs1

    def train_layer2(self, batch, is_validation_step=False):
        #create input layer
        input_layer, weight = self.get_input_layer(batch)
        if self.layer1_enc is None:
            print("Please train layer 1 first")
            return

        layer1_enc_out = self.layer1_enc(input_layer)
        layer2_input = self.get_input_layer2(layer1_enc_out)
        if self.layer2_enc is None:
            self.set_up_layer2(layer2_input)

        with tf.GradientTape() as tape:
            layer2_enc_out = self.layer2_enc(layer2_input)
            layer2_dec_out = self.layer2_dec(layer2_enc_out)
            layer1_dec_input = tf.reshape(layer2_dec_out, tf.shape(layer1_enc_out))
            layer1_dec_out = self.layer1_dec(layer1_dec_input)
            loss = tf.keras.losses.MSE( input_layer, layer1_dec_out)
            loss = loss * tf.add(tf.dtypes.cast(weight, tf.float32), tf.constant(1.0))
            if not is_validation_step:
                grads = tape.gradient(loss, self.layer2_enc.trainable_variables + self.layer2_dec.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.layer2_enc.trainable_variables +
                                                    self.layer2_dec.trainable_variables))
        return np.sum(loss), layer1_dec_out

    def train_layer3(self, batch, is_validation_step=False):
        #create input layer
        input_layer, weight = self.get_input_layer(batch)
        if self.layer2_enc is None:
            print("Please train layer 2 first")
            return

        layer1_enc_out = self.layer1_enc(input_layer)
        layer2_input = self.get_input_layer2(layer1_enc_out)
        layer2_enc_out = self.layer2_enc(layer2_input)
        layer3_input = self.get_input_layer3(layer2_enc_out)

        if self.layer3_enc is None:
            self.set_up_layer3(layer3_input)

        with tf.GradientTape() as tape:
            layer3_enc_out = self.layer3_enc(layer3_input)
            layer3_dec_out = self.layer3_dec(layer3_enc_out)
            layer2_dec_input = tf.reshape(layer3_dec_out, tf.shape(layer2_enc_out))
            layer2_dec_out = self.layer2_dec(layer2_dec_input)
            layer1_dec_input = tf.reshape(layer2_dec_out, tf.shape(layer1_enc_out))
            layer1_dec_out = self.layer1_dec(layer1_dec_input)
            loss =  tf.keras.losses.MSE( input_layer, layer1_dec_out )
            loss = loss * tf.add(tf.dtypes.cast(weight, tf.float32), tf.constant(1.0))
            if not is_validation_step:
                grads = tape.gradient(loss, self.layer3_enc.trainable_variables + self.layer3_dec.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.layer3_enc.trainable_variables + \
                                                    self.layer3_dec.trainable_variables))
        return np.sum(loss), layer1_dec_out

    def train_layer4(self, batch, is_validation_step=False):
        #create input layer
        input_layer, weight = self.get_input_layer(batch)
        if self.layer3_enc is None:
            print("Please train layer 3 first")
            return

        layer1_enc_out = self.layer1_enc(input_layer)
        layer2_input = self.get_input_layer2(layer1_enc_out)
        layer2_enc_out = self.layer2_enc(layer2_input)
        layer3_input = self.get_input_layer3(layer2_enc_out)
        layer3_enc_out = self.layer3_enc(layer3_input)
        layer4_input = self.get_input_layer4(layer3_enc_out)

        if self.layer4_enc is None:
            self.set_up_layer4(layer4_input)

        with tf.GradientTape() as tape:
            layer4_enc_out = self.layer4_enc(layer4_input)
            layer4_dec_out = self.layer4_dec(layer4_enc_out)
            layer3_dec_input = tf.reshape(layer4_dec_out, tf.shape(layer3_enc_out))
            layer3_dec_out = self.layer3_dec(layer3_dec_input)
            layer2_dec_input = tf.reshape(layer3_dec_out, tf.shape(layer2_enc_out))
            layer2_dec_out = self.layer2_dec(layer2_dec_input)
            layer1_dec_input = tf.reshape(layer2_dec_out, tf.shape(layer1_enc_out))
            layer1_dec_out = self.layer1_dec(layer1_dec_input)
            loss = tf.keras.losses.MSE( input_layer, layer1_dec_out )
            loss = loss * tf.add(tf.dtypes.cast(weight, tf.float32), tf.constant(1.0))
            if not is_validation_step:
                grads = tape.gradient(loss, self.layer4_enc.trainable_variables + self.layer4_dec.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.layer4_enc.trainable_variables + \
                                                self.layer4_dec.trainable_variables))
        return np.sum(loss), layer1_dec_out

    def train_all_layers(self, batch, is_validation_step=False):
        #create input layer
        input_layer, weight = self.get_input_layer(batch)
        if self.layer4_enc is None:
            print("Please train layer 4 first")
            return

        with tf.GradientTape() as tape:
            layer1_enc_out = self.layer1_enc(input_layer)
            layer2_input = self.get_input_layer2(layer1_enc_out)
            layer2_enc_out = self.layer2_enc(layer2_input)
            layer3_input = self.get_input_layer3(layer2_enc_out)
            layer3_enc_out = self.layer3_enc(layer3_input)
            layer4_input = self.get_input_layer4(layer3_enc_out)
            layer4_enc_out = self.layer4_enc(layer4_input)
            layer4_dec_out = self.layer4_dec(layer4_enc_out)
            layer3_dec_input = tf.reshape(layer4_dec_out, tf.shape(layer3_enc_out))
            layer3_dec_out = self.layer3_dec(layer3_dec_input)
            layer2_dec_input = tf.reshape(layer3_dec_out, tf.shape(layer2_enc_out))
            layer2_dec_out = self.layer2_dec(layer2_dec_input)
            layer1_dec_input = tf.reshape(layer2_dec_out, tf.shape(layer1_enc_out))
            layer1_dec_out = self.layer1_dec(layer1_dec_input)
            loss = tf.keras.losses.MSE( input_layer, layer1_dec_out )
            loss = loss * tf.dtypes.cast(weight, tf.float32)
            if not is_validation_step:
                grads = tape.gradient(loss,
                                      self.layer1_enc.trainable_variables + self.layer1_dec.trainable_variables +
                                      self.layer2_enc.trainable_variables + self.layer2_dec.trainable_variables +
                                      self.layer3_enc.trainable_variables + self.layer3_dec.trainable_variables +
                                      self.layer4_enc.trainable_variables + self.layer4_dec.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,
                                       self.layer1_enc.trainable_variables + self.layer1_dec.trainable_variables +
                                       self.layer2_enc.trainable_variables + self.layer2_dec.trainable_variables +
                                       self.layer3_enc.trainable_variables + self.layer3_dec.trainable_variables +
                                       self.layer4_enc.trainable_variables + self.layer4_dec.trainable_variables))
        return np.sum(loss), self.get_embedding(layer4_enc_out)

    def calculate_embedding(self, batch):
        input_layer, _ = self.get_input_layer(batch)
        if self.layer4_enc is None:
            print("Please train layer 4 first")
            return

        layer1_enc_out = self.layer1_enc(input_layer)
        layer2_input = self.get_input_layer2(layer1_enc_out)
        layer2_enc_out = self.layer2_enc(layer2_input)
        layer3_input = self.get_input_layer3(layer2_enc_out)
        layer3_enc_out = self.layer3_enc(layer3_input)
        layer4_input = self.get_input_layer4(layer3_enc_out)
        layer4_enc_out = self.layer4_enc(layer4_input)

        id = tf.reshape(batch, (tf.shape(batch)[0],1))
        embedding = np.hstack([id, self.get_embedding(layer4_enc_out)])
        return embedding

    def set_constant_data(self, features, in_sample, out_sample, in_sample_amnt, out_sample_amnt):

        self.features = tf.constant(features, name="features")
        self.in_sample = tf.constant(in_sample, dtype=tf.int64, name="in_sample")
        self.out_sample = tf.constant(out_sample, dtype=tf.int64, name="out_sample")
        self.in_sample_amnt = tf.constant(in_sample_amnt, name="in_sample_amnt")
        self.out_sample_amnt = tf.constant(out_sample_amnt, name="out_sample_amnt")

    def reset_layer(self, layer):
        if layer == 1: self.layer1_enc = None
        if layer == 2: self.layer2_enc = None
        if layer == 3: self.layer3_enc = None
        if layer == 4: self.layer4_enc = None