#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:22:45 2019

@author: tonpoppe
"""

import random
import tensorflow as tf
import networkx as nx
import datetime
import numpy as np
from tensorflow import keras


class DataFeederNx:
    """
    This class reads a directed network object and covert this to
    the samples for training the GraphCase algorithm.
    Note that the first label with index 0 is used as edge weight.
    No multiple edge labels are currently supported.
    """
    DUMMY_WEIGHT = 0  # weigh assigned to a dummy edge

    def __init__(
        self, graph, neighb_size=3, batch_size=3, val_fraction=0.3, verbose=False, 
        seed=1, weight_label='weight', encoder_labels=None):
        #TODO set custom dummy node
        self.__check_graph(graph)
        self.val_frac = val_fraction  # fraction of nodes used for validation set
        self.batch_size = batch_size  # number of nodes in a training batch
        self.neighb_size = neighb_size  # size of the neighborhood sampling
        self.graph = graph
        self.verbose = verbose
        self.weight_label = weight_label
        self.encoder_labels = encoder_labels  # labels used for training the encoder
        self.iter = {}
        self.feature_dim = 0
        self.used_encoder_labels=None
        self.lbls=None  # vector containing the node labels for supervised training
        self.features = self.__extract_features(encoder_labels)
        self.edge_labels = self.__get_valid_edge_labels()
        self.in_sample, self.in_sample_weight = self.__extract_in_sample()
        self.out_sample, self.out_sample_weight = self.__extract_out_sample()
        self.seed = seed
        # self.edge_labels = None
        self.train_epoch_size = 0
        self.val_epoch_size = 0
        self.label_name = None  #label used for supervised training


    def init_train_batch(self, label_name=None):
        """
        Creates a training batch and validation batch based on a random split of the nodes
        in the graph. The batches contain a list of node id's only and are assigned to
        the iter attribute.
        """
        # split nodes in train set
        self.label_name = label_name
        if label_name:
            assert len([l for l in self.used_encoder_labels if l==label_name])==0, \
                'label name is also used a node attribute'
            #retrieve the labels
            self.__extract_labels(label_name)
        train, val = self.__train_test_split()
        self.train_epoch_size = len(train)
        self.val_epoch_size= len(val)
        if self.verbose:
            print(f"train nodes {train[:10]} ...")
            print(f"val nodes {val[:10]} ...")
        self.iter["train"] = self.create_sample_iterators(train, self.batch_size)
        self.iter["valid"] = self.create_sample_iterators(val, self.batch_size)

    def __train_test_split(self):
        """
        Split the nodes in the graph in a train and test set.
        The fraction of nodes assigned to the test set is based on the
        val_frac parameter.

        @return tupple with a list of the train and test nodes.
        """
        nodes = list(self.graph.nodes)
        random.Random(self.seed).shuffle(nodes)
        split_value = round(self.val_frac * len(nodes))
        test_data = nodes[:split_value]
        train_data = nodes[split_value:]
        return train_data, test_data

    def __extract_features(self, encoder_labels=None):
        """
        Creates a feature matrix based on the node labels in the graph.
        Only node labels which are populated for all nodes are included.

        returns:    2d numpy array of features.
        """
        lbls = self.__get_valid_node_labels(encoder_labels=encoder_labels)

        features = []
        for lbl in lbls:
            features.append([x for _, x in \
                            sorted(nx.get_node_attributes(self.graph, lbl).items())])

        #append dummy node
        for feature in features:
            feature.append(0)

        assert len(features[0]) == len(list(self.graph.nodes))+1, \
               "number of features deviates from number of nodes"

        return np.array(features).transpose().astype(np.float32)

    def __extract_out_sample(self):
        """
        Extracts the  deterministic sampling of size equal to neighb_size
        for the outgoing neighbourhood. The deterministic sampling in based
        on edge weight in order from high to low

        @return a tuple with the first element a 2d numpy with the outgoing
                adjacent node ids of the sample per node. The second element
                is a 2d numpy array containing the edge weights of the sample
                per node
        """
        out_edges_dict = {}
        for out_node, in_node, weight in self.graph.out_edges(data=True):
            out_edges_dict[out_node] = out_edges_dict.get(out_node, list()) + \
                                       [(in_node, [weight[lbl] for lbl in self.edge_labels])]
            # out_weight_dict = out_weight_dict.get(out_node, list()) + \
            #                            [(in_node, weight[self.weight_label])]

        return self.__convert_dict_to_node_and_weight_list(out_edges_dict)

    def __extract_in_sample(self):
        """
        Extracts the  deterministic sampling of size equal to neighb_size
        for the incoming neighbourhood. The deterministic sampling in based
        on edge weight in order from high to low

        @return a tuple with the first element a 2d numpy with the incoming
                adjacent node ids of the sample per node. The second element
                is a 3d numpy array containing the edge weights of the sample
                per node.
        """
        in_edges_dict = {}
        for out_node, in_node, weight in self.graph.in_edges(data=True):
            in_edges_dict[in_node] = in_edges_dict.get(in_node, list()) + \
                                    [(out_node, [weight[lbl] for lbl in self.edge_labels])]

        return self.__convert_dict_to_node_and_weight_list(in_edges_dict)

    def __convert_dict_to_node_and_weight_list(self, edges_dict):
        """
        Helper function that converts a dictionary with tuples of neighbours
        and weight into a nodes sample and weight sample of size equal to
        neighb_size based on a sorted weight. if required the neighbouring
        nodes are extended with dummy nodes up to the neighb_size.

        Args:
            edges_dict: dictionary with key the node id and value a list of tupples with the
                    neighbouring node ids and weights in an unsorted order.
        """
        dummy_id = self.features.shape[0]-1
        dummy_lbl = [DataFeederNx.DUMMY_WEIGHT] * len(self.edge_labels)
        nodes = list(self.graph.nodes)
        for node in nodes:
            # sort neighbours by weight
            neighbours = sorted(edges_dict.get(node, [(dummy_id, dummy_lbl)]),
                                key=lambda x: x[1][0], reverse=True)

            if len(neighbours) <= self.neighb_size:
                neighbours = neighbours + [(dummy_id, dummy_lbl)] * \
                             (self.neighb_size - len(neighbours))
            else:
                neighbours = neighbours[0:self.neighb_size]
            edges_dict[node] = neighbours

        edges_list = []
        weight_list = []
        for  _, neighbours in sorted(edges_dict.items()):
            edges_list.append([t[0] for t in neighbours])
            weight_list.append([t[1] for t in neighbours])

        # add dummy node
        edges_list.append([dummy_id] * self.neighb_size)
        weight_list.append([dummy_lbl] * self.neighb_size)

        return (np.array(edges_list), np.array(weight_list).astype(np.float32))

    def __get_valid_edge_labels(self):
        """
        Checks which edge labels are set for all edges. Labels which are only set for part of the
        edges are excluded as lables, all others are included as edge labels.
        The edge labels used for edge weight is set in front of the list.

        Returns:
            list of valid edge labels.
        """
        edge_label_stats = {}
        for _, _, lbls in self.graph.in_edges(data=True):
            for lbl in lbls.keys():
                edge_label_stats[lbl] = edge_label_stats.get(lbl, 0) + 1

        max_value = max(edge_label_stats.values())
        incl_list = []
        excl_list = []
        for lbl, cnt in edge_label_stats.items():
            if cnt == max_value:
                incl_list.append(lbl)
            else:
                excl_list.append(lbl)

        # set edge weight as first item in the list
        try:
            incl_list.remove(self.weight_label)
        except ValueError:
            print(f"ERROR {self.weight_label} doesn't seem to be a valid label")
            exit()
        incl_list = [self.weight_label] + incl_list

        if self.verbose:
            print(f"The following edge labels are excluded {excl_list}")
            print(f"The following edge labels are included {incl_list}")
        return incl_list

    def __get_valid_node_labels(self, encoder_labels=None):
        """
        Checks which node labels are set for all nodes. Labels which are only
        populated for a limited part of the nodes in the graph are excluded as
        labels, all others are included as node features.
        """
        node_label_stats = {}  # keeps track if the node label is available for all nodes
        for i in self.graph.nodes.keys():
            for lbl in  self.graph.nodes[i].keys():
                node_label_stats[lbl] = node_label_stats.get(lbl, 0) + 1

        max_value = max(node_label_stats.values())
        incl_list = []
        excl_list = []
        for lbl, cnt in node_label_stats.items():
            if cnt == max_value:
                incl_list.append(lbl)
            else:
                excl_list.append(lbl)

        if encoder_labels:
            incl_list = [l for l in incl_list if l in encoder_labels]
            not_found = [l for l in encoder_labels if l not in incl_list]
            assert len(not_found) == 0, \
                f"The follwing encoders labels could not be found {not_found}"
        self.used_encoder_labels = incl_list
        self.feature_dim = len(incl_list)
        if self.verbose:
            print(f"The following node labels are excluded {excl_list}")
            print(f"The following node labels are included {incl_list}")
        return incl_list

    def __extract_labels(self, label_name):
        """retrieves the labels from the graph."""
        self.lbls = tf.constant([x for _, x in sorted(self.graph.nodes(label_name, 0))])

    def init_incr_batch(self, nodes=None):
        """
        Creates an dataset iterator containing all nodes in the dataset
        @param nodes a list of nodes that need to be included in the dataset
                    if not populated then all nodes of the graph are taken.

        @return tf batches dataset
        """
        if not nodes:
            nodes = list(self.graph.nodes)

        #amend list with dummy nodes to make len is mulitple of the size.
        dummy_id = self.features.shape[0]-1
        if len(nodes) % self.batch_size != 0:
            nodes = nodes + [dummy_id] * (self.batch_size - (len(nodes) % self.batch_size))

        dataset = tf.data.Dataset.from_tensor_slices(nodes)
        batched_dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return batched_dataset

    def __check_graph(self, graph):
        '''
        Check if the graph is directed.
        '''
        assert nx.is_directed(graph), "Only Directed graph are currently supported"

    def get_train_samples(self):
        """
        Returns the training dataset iterator
        """
        return self.iter["train"]

    def get_val_samples(self):
        """
        Returns the validation dataset iterator
        """
        return self.iter["valid"]

    def create_sample_iterators(self, node_list, size):
        """
        Converts a list of nodes into an tensorflow dataset for training.

        :param node_list: list of nodes
        :param size: The size of the batches.
        :return: tensorflow  dataset of node list of size size
        """
        dataset = tf.data.Dataset.from_tensor_slices(node_list)
        batched_dataset = dataset.repeat(count=-1).batch(size, drop_remainder=False)
        return batched_dataset


    def get_feature_size(self):
        """
        Returns the length of node + egdel labels.
        """
        return self.feature_dim + len(self.edge_labels)

    def get_number_of_node_labels(self):
        return self.feature_dim

        
    def get_features(self, node_id):
        """
        Retrieve the node labels of the node with id = node_id

        Args:
            node_id: id of node for which the features are retrieved

        returns:
            numpy array with the node lables.
        """
        return self.features[node_id]
