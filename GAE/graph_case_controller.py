#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 07:26:20 2019

@author: tonpoppe
"""
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from GAE.model import GraphAutoEncoderModel
from GAE.data_feeder_nx import DataFeederNx
from GAE.graph_reconstructor import GraphReconstructor


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
                 seed=1,
                 weight_label='weight',
                 act=tf.nn.sigmoid,
                 useBN=False
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
        self.act = act
        self.weight_label = weight_label
        self.useBN = useBN

        self.__consistency_checks()
        self.sampler = self.__init_datafeeder_nx()
        self.model = self.__init_model()

    def __init_datafeeder_nx(self):
        """
        Initialises the datafeeder
        """
        sampler = DataFeederNx(self.graph, neighb_size=max(self.support_size),
                               batch_size=self.batch_size, verbose=self.verbose, seed=self.seed,
                               weight_label=self.weight_label)
        return sampler

    def __init_model(self):
        """
        Initialises the model
        """
        model = GraphAutoEncoderModel(self.learning_rate,
                                      self.dims,
                                      self.support_size,
                                      verbose=self.verbose,
                                      seed=self.seed,
                                      dropout=None,
                                      act=self.act,
                                      useBN=self.useBN)

        # set feature file and in and out samples
        features = self.sampler.features
        in_sample = self.sampler.in_sample
        out_sample = self.sampler.out_sample
        in_sample_amnt = self.sampler.in_sample_weight
        out_sample_amnt = self.sampler.out_sample_weight
        model.set_constant_data(features, in_sample, out_sample,
                                in_sample_amnt, out_sample_amnt)
        return model


    def train_layer(self, layer, all_layers=False, dim=None, learning_rate=None, act="pass",
                    dropout=None, steps=None):
        """
        Trains a specific layer of the model. Layer need to be trained from bottom
        to top, i.e. layer 1 to the highest layer.

        args:
            layer:  Number of the layer to be trained. This layer will be reset before
                    training.
            all_layers: Boolean indicating if all layers need to be trained
                    together. the specified layer is then not reset.
            dim:    Dimension to be used for the layer. This will overwrite
                    the dimension set during initialisation and can typically
                    be used for a layer wise hyper parameter search. This
                    is only applied on the specified layer.
            learning_rate: The learning rate to be used for the layer. This will
                    overwrite the learning rate set during initialisation. TThis
                    is only applied on the specified layer.
            act:    The activation function used for the layer. This
                    is only applied on the specified layer.

        Returns:
            A dictionary with the validation information of all validation
            batches.
        """
        if self.verbose:
            print(f"Training layer {layer}")

        if act == "pass":
            act = self.act
        if dim is None:
            dim = self.dims[layer-1]
        if learning_rate is None:
            learning_rate = self.learning_rate
        if steps is None:
            steps = self.max_total_steps

        if not all_layers:
            self.model.set_hyperparam(layer, dim, act, learning_rate, dropout)
            self.model.reset_layer(layer)

        self.sampler.init_train_batch()
        self.__init_history()
        counter = 0
        for i in self.sampler.get_train_samples():
            try:
                train_loss, _ = self.model.train_layer(layer, i, all_layers=all_layers)

                # validation & print step
                if counter % self.validate_iter == 0:
                    val_counter = 0
                    val_loss = 0
                    for j in self.sampler.get_val_samples():
                        val_l, _ = self.model.train_layer(layer, j, is_validation_step=True)
                        val_loss += val_l
                        val_counter += 1
                        if val_counter == 10:
                            break

                    val_loss = val_loss / val_counter
                    # Print results
                    if self.verbose:
                        print("layer", layer, "-", all_layers,
                              "Iter:", '%04d' % counter,
                              "train_loss=", "{:.5f}".format(train_loss),
                              "val_loss=", "{:.5f}".format(val_loss),
                              "time=", time.strftime('%Y-%m-%d %H:%M:%S'))
                    self.__update_history(counter, train_loss, val_loss,
                                          time.strftime('%Y-%m-%d %H:%M:%S'))

                counter += 1
                if counter == steps:
                    break

            except tf.errors.OutOfRangeError:
                print("reached end of batch via out of range error")
                break

        return self.history

    def calculate_embeddings(self, graph=None, nodes=None):
        """
        Calculated the embedding of the nodes specified. If no nodes are
        specified, then the embedding for all nodes are calculated.

        Args:
            graph:  Optionally the graph for which the embeddings need to be calculated. If set to
                    None then the graph used for initializing is used.
            nodes:  Optionally a list of node ids in the graph for which the
                    embedding needs to be calculated.

        Returns:
            A 2d numpy array with one embedding per row.
        """
        print("calculating all embeddings")
        if graph is not None:
            self.graph = graph
            self.__init_datafeeder_nx()

        embedding = None
        counter = 0
        for i in self.sampler.init_incr_batch(nodes):
            counter += 1
            try:
                embed = self.model.calculate_embedding(i)
                if embedding is None:
                    embedding = embed
                else:
                    embedding = np.vstack([embedding, embed])

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

    def __update_history(self, i, train_l, val_l, curtime):
        """
        Adds the information of a validation batch to the history dict.
        """
        self.history["i"].append(i)
        self.history["l"].append(train_l)
        self.history["val_l"].append(val_l)
        self.history["time"].append(curtime)

    def save_model(self, save_path):
        """
        Saves the layers of the trained model. Every layer is stored in a seperate file.

        Args:
            save_path: path in which the layers are stored.
        """
        self.model.save_weights(save_path)

    def load_model(self, filename, graph):
        """
        Loads a trained model from a pickle file

        Args:
            filename: filename of the pickle with the stored model.
        """
        self.fit(graph, verbose=False, steps=1)
        self.model.load_weights(filename)

    def __consistency_checks(self):
        """
        Performs the following consistency checks.
        1) len of dims list is 2 * len support size or len is 2 * support size + 1
        """
        assert len(self.dims) == 2 * len(self.support_size) or \
               len(self.dims) -1 == 2 * len(self.support_size), \
               f"number of dims {len(self.dims)} does not match with two times the number of " \
               f"support sizes {len(self.support_size)}"

    def fit(self, graph=None, verbose=None, steps=None):
        if verbose is not None:
            self.verbose = verbose
        
        if graph is not None:
            self.graph = graph
            self.__init_datafeeder_nx()

        train_res = {}
        for i in range(len(self.dims)):
            train_res["l"+str(i+1)] = self.train_layer(i+1, steps=steps)

        train_res['all'] = self.train_layer(len(self.dims), all_layers=True, steps=steps)
        return train_res

    def get_l1_structure(self, node_id, graph=None, verbose=None, show_graph=False,
                         node_label=None, get_pyvis=False):
        """
        Retrieve the input layer and corresponding sampled graph of the local neighbourhood.

        Args:
            node_id:    id of the node for which the input layer is calculated
            graph:      graph to sample. If no graph is specified then the current graph is used.
            show_graph  Boolean indicating if a plot of the graph needs to be generated.
            node_label  Label used for the nodes. If None then the node id is used.

        returns:
            a networkx graph of the sampled neighbourhood and a numpy matrix of the input layer.
        """
        if verbose is not None:
            self.verbose = verbose
        
        if graph is not None:
            self.graph = graph
            self.__init_datafeeder_nx()

        inputlayer, _ = self.model.get_input_layer([node_id], hub=1)
        target = self.sampler.get_features(node_id)
        graph_rec = GraphReconstructor()
        recon_graph = graph_rec.reconstruct_graph(target, inputlayer, self.support_size)

        if show_graph:
            graph_rec.show_graph(recon_graph, node_label=node_label)

        if get_pyvis:
            nt = graph_rec.show_pyvis(recon_graph, node_label=node_label)
            return inputlayer, recon_graph, nt

        return inputlayer, recon_graph
