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
from GAE.model_copy import GraphAutoEncoderModel
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
                 hub0_feature_with_neighb_dim=None,
                 batch_size=3,
                 verbose=False,            
                 seed=1,
                 weight_label='weight',
                 act=tf.nn.sigmoid,
                 useBN=False
                 ):
        self.graph = graph
        self.learning_rate = learning_rate
        self.dims = dims
        self.hub0_feature_with_neighb_dim = hub0_feature_with_neighb_dim
        self.batch_size = batch_size
        self.support_size = support_size
        self.verbose = verbose
        self.seed = seed
        self.act = act
        self.weight_label = weight_label
        self.useBN = useBN

        self.__consistency_checks()
        self.__init_datafeeder_nx()
        self.model = self.__init_model()
        self.__set_dataset()

    def __init_datafeeder_nx(self):
        """
        Initialises the datafeeder
        """
        sampler = DataFeederNx(self.graph, neighb_size=max(self.support_size),
                               batch_size=self.batch_size, verbose=self.verbose, seed=self.seed,
                               weight_label=self.weight_label)
        self.sampler = sampler

    def __init_model(self):
        """
        Initialises the model
        """
        model = GraphAutoEncoderModel(
            self.dims,
            self.support_size,
            self.sampler.get_feature_size(),
            hub0_feature_with_neighb_dim=self.hub0_feature_with_neighb_dim,
            number_of_node_labels=self.sampler.get_number_of_node_labels(),
            verbose=self.verbose,
            seed=self.seed,
            dropout=None,
            act=self.act,
            useBN=self.useBN)

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate)        
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def __set_dataset(self):
        # set feature file and in and out samples
        features = self.sampler.features
        in_sample = self.sampler.in_sample
        out_sample = self.sampler.out_sample
        in_sample_amnt = self.sampler.in_sample_weight
        out_sample_amnt = self.sampler.out_sample_weight
        self.model.set_constant_data(features, in_sample, out_sample,
                                     in_sample_amnt, out_sample_amnt)

        # below code is required to build the model as it won't build on the 
        # dataset
        self.sampler.init_train_batch()
        train_data = self.sampler.get_train_samples()
        for n in train_data.take(1):
            x, w = self.model.get_input_layer(n, hub=1)
            features = self.model.get_features(n)
            self.model((features, x))

    def calculate_embeddings(self, graph=None, nodes=None, verbose=False):
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
        self.verbose = verbose
        if verbose:
            print("calculating all embeddings")
        
        if graph is not None:
            self.graph = graph
            self.__init_datafeeder_nx()
            self.__set_dataset()

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

        if verbose:
            print("reached end of batch")
        return embedding

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

    def fit(self, graph=None, verbose=None, epochs=4):
        if verbose is not None:
            self.verbose = verbose
        
        if graph is not None:
            self.graph = graph
            self.__init_datafeeder_nx()
            self.__set_dataset()

        self.sampler.init_train_batch()
        train_data = self.sampler.get_train_samples()
        train_data = train_data.map(lambda x: (self.model.get_features(x), self.model.get_input_layer(x, hub=1)))
        train_data = train_data.map(lambda x, i: ((x, i[0]), (x, i[0])))
        # validation_data = self.sampler.get_val_samples()
        # validation_data = validation_data.map(lambda x: (x, x))

        steps_per_epoch = int(self.sampler.train_epoch_size / self.batch_size)
        validation_steps = int(self.sampler.val_epoch_size / self.batch_size)
        history = self.model.fit(
            train_data, 
            # validation_data=validation_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # validation_steps = validation_steps
        )  
        return history

    def get_l1_structure(self, node_id, graph=None, verbose=None, show_graph=False,
                         node_label=None, get_pyvis=False):
        """
        Retrieve the input layer and corresponding sampled graph of the local neighbourhood.

        Args:
            node_id:    id of the node for which the input layer is calculated
            graph:      graph used for sampling. If no graph is specified then the current graph is used.
            show_graph  Boolean indicating if a plot of the graph needs to be generated.
            node_label  Label used for plotting the nodes. If None then the node id is used.

        returns:
            a networkx graph of the sampled neighbourhood and a numpy matrix of the input layer.
        """
        if verbose is not None:
            self.verbose = verbose
    
        if graph is not None:
            self.graph = graph
            self.__init_datafeeder_nx()
            self.__set_dataset()

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

    def decode(self, embedding, incl_graph=None):
        """
        Decodes the given embedding into a node and local neighbourhood.
        Args:
            embedding   : Embedding of the node
            incl_graph  : {None | nx | pyvis | graph }

        Returns:
            A tuple with the node labels, inputlayer and optionally a graph.
        """
        feat_out, df_out = self.model.decode(embedding)
        if incl_graph is not None:
            graph_rec = GraphReconstructor()
            recon_graph = graph_rec.reconstruct_graph(feat_out, df_out, self.support_size)
        
        if incl_graph == 'graph':
            return feat_out, df_out, recon_graph

        if incl_graph == 'pyvis':
            nt = graph_rec.show_pyvis(recon_graph)
            return feat_out, df_out, nt

        return feat_out, df_out
