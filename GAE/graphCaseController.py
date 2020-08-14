#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 07:26:20 2019

@author: tonpoppe
"""

import time
import numpy as np
import pandas as pd
import math
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import pickle
import networkx as nx
from GAE.model import GraphAutoEncoderModel
from GAE.dataFeederNx import DataFeederNx


class GraphAutoEncoder:
    """
    This class implement the graphCase algorithm. Refer for more details 
    to the corresponding documentation.  

    Args:
        graph:      graph on which the embedding is trained. Only bi-directed 
                    graphs re supported.
        learning_rate: learning rate of the MLP.
        support_size: list with number of sampled edges per layer. The
                    current implementation only support one size for all layers
        dims:       list with the dimension per layer.
        batch_size: number of nodes per training cycle.
        max_total_steps: Number of batches used for training the mlp.
        validate_iter: Number of batches between a validation batch.
        verbose:    boolean if True then detailed feedback on the training progress
                    is given.
        seed:       Seed used for the random split in train and test set.

    """
    def __init__(self,
                graph, 
                 learning_rate=0.0001,
                 support_size=[2, 2],
                 dims=[32, 32, 32, 32],  
                 batch_size=3,
                 max_total_steps=100,
                 validate_iter=5,
                 verbose=False,
                 seed=1
                 ):
        self.graph = graph
        self.max_total_steps = max_total_steps
        self.validate_iter = validate_iter
        self.learning_rate = learning_rate
        self.history = {}
        self.dims = dims
        self.batch_size = batch_size
        self.support_size = support_size
        self.verbose = verbose
        self.seed = seed

        self.__init_datafeeder_nx()
        self.__init_model()
    

    def __init_datafeeder_nx(self):
        """
        Initialises the datafeeder
        """
        self.sampler = DataFeederNx(self.graph, neighb_size=max(self.support_size), batch_size=self.batch_size,
                                    verbose=self.verbose, seed=self.seed)
        self.feature_size = self.sampler.get_feature_size()


    def __init_model(self):  
        """
        Initialises the model
        """
        self.model = GraphAutoEncoderModel(self.learning_rate,
                                        self.dims,
                                        self.feature_size,
                                        verbose=self.verbose)

        # set feature file and in and out samples
        features= self.sampler.features
        in_sample = self.sampler.in_sample
        out_sample = self.sampler.out_sample
        in_sample_amnt = self.sampler.in_sample_weight
        out_sample_amnt = self.sampler.out_sample_weight
        self.model.set_constant_data( features, in_sample, out_sample, in_sample_amnt, out_sample_amnt)


    def train_layer(self, layer, dim=None , learning_rate=None, act=tf.nn.relu):  
        """
        Trains a specific layer of the model. Layer need to be trained from bottom
        to top, i.e. layer 1 to the highest layer.

        args:
            layer:  Number of the layer. If all layers need to be trained 
                    together then the keyword 'all' can be used.
            dim:    Dimension to be used for the layer. This will overwrite
                    the dimension set during initialisation and can typically
                    be used for a layer wise hyper parameter search. This 
                    is only available for single layers.
            learning_rate: The learning rate to be used for the layer. This will
                    overwrite the learning rate set during initialisation. This 
                    is only available for single layers.
            act:    The activation function used for the layer. This 
                    is only available for single layers.

        Returns:
            A dictionary with the validation information of all validation 
            batches.
        """
        if self.verbose:
            print(f"Training layer {layer}")      
        if layer == 'all':
            method = "train_all_layers"
        else:
            method = 'train_layer' + str(layer)
            if dim is None:
                dim = self.dims[layer-1]
            if learning_rate is None:
                learning_rate = self.learning_rate
            self.model.set_hyperparam(layer, dim,act,learning_rate)
            self.model.reset_layer(layer)

        self.sampler.init_train_batch()
        self.__init_history()
        counter = 0
        for i in self.sampler.get_train_samples():
            try:
                l, _ = getattr(self.model, method)(i)

                # validation & print step
                if counter % self.validate_iter == 0:
                    val_counter = 0
                    val_loss = 0
                    for j in self.sampler.get_val_samples():
                        val_l, _ = getattr(self.model, method)(j, is_validation_step=True)
                        val_loss += val_l
                        val_counter += 1
                        if val_counter == 10:
                            break

                    val_loss = val_loss / val_counter
                    # Print results
                    if self.verbose:
                        print("Iter:", '%04d' % counter,
                            "train_loss=", "{:.5f}".format(l),
                            "val_loss=", "{:.5f}".format(val_loss),
                            "time=", time.strftime('%Y-%m-%d %H:%M:%S'))
                    self.__update_history(counter, l, val_loss, time.strftime('%Y-%m-%d %H:%M:%S'))

                counter += 1
                if counter == self.max_total_steps:
                    break

            except tf.errors.OutOfRangeError:
                print("reached end of batch via out of range error")
                break

        return self.history

    def calculate_embeddings(self, nodes=None):
        """
        Calculated the embedding of the nodes specified. If no nodes are
        specified, then the embedding for all nodes are calculated.

        Args:
            nodes:  Optionally a list of node ids in the graph for which the
                    embedding needs to be calculated.

        Returns:    
            A 2d numpy array with one embedding per row.
        """
        print("calculating all embeddings")

        embedding = None
        counter = 0
        for i in self.sampler.init_incr_batch(nodes):
            counter += 1
            try:
                e = self.model.calculate_embedding(i)
                if embedding is None:
                    embedding = e
                else:
                    embedding = np.vstack([embedding,e])

                if counter % 100 == 0:
                    print("processed ", counter, " batches time: ", datetime.now())

            except tf.errors.OutOfRangeError:
                break

        print("reached end of batch")
        return embedding


    def __init_history(self):
        """
        Initialises a dictionary containing for capturing information from the
        validation batches.
        """
        self.history = {}
        self.history["i"] = []
        self.history["l"] = []
        self.history["val_l"] = []
        self.history["time"] = []

    def __update_history(self, i, l, val_l, curtime):
        """
        Adds the information of a validation batch to the history dict.
        """
        self.history["i"].append(i)
        self.history["l"].append(l)
        self.history["val_l"].append(val_l)
        self.history["time"].append(curtime)

    def save_model(self, filename):
        """
        Saves a trained model in a pickle file

        Args:
            filename: filename of the pickle to which the model is stored.
        """
        pickle.dump(self.model, open(filename, "wb"))

    def load_model(self, filename):
        """
        Loads a trained model from a pickle file

        Args:
            filename: filename of the pickle with the stored model.
        """
        self.model = pickle.load(open(filename, "rb"))