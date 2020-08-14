#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:22:45 2019

@author: tonpoppe
"""

import os
import random
import tensorflow as tf
from tensorflow import keras
import networkx as nx
import datetime
from multiprocessing.pool import ThreadPool
import numpy as np
from pathlib import Path
import csv

# MAC OS bug
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class DataFeederNx:
    """
    This class reads a directed network object and covert this to 
    the samples for training the GraphCase algorithm.
    Note that the first label with index 0 is used as edge weight. 
    No multiple edge labels are currently supported.
    """
    DUMMY_WEIGHT = 0  # weigh assigned to a dummy edge

    def __init__(self, graph, neighb_size=3, batch_size=3, val_fraction=0.3, 
                verbose=False, seed=1):
        #TODO check if graph is bi-directed, set custom dummy node
        self.val_frac = val_fraction  # fraction of nodes used for validation set
        self.batch_size = batch_size  # number of nodes in a training batch
        self.neighb_size = neighb_size  # size of the neighborhood sampling
        self.graph = graph
        self.verbose=verbose
        self.iter = {}
        self.features = self.__extract_features()
        self.in_sample, self.in_sample_weight = self.__extract_in_sample()
        self.out_sample, self.out_sample_weight = self.__extract_out_sample()
        self.seed = seed


    def init_train_batch(self):
        """
        Creates a training batch and validation batch based on a random split of the nodes
        in the graph. The batches contain a list of node id's only and are assigned to 
        the iter attribute.
        """
        # split nodes in train set
        train, val = self.__train_test_split()
        if self.verbose:
            print(f"train nodes {train}")
            print(f"val nodes {val}")
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

    def __extract_features(self):
        """
        Creates a feature matrix based on the node labels in the graph.
        Only node labels which are populated for all nodes are included.

        @return 2d numpy array of features.
        """
        lbls = self.__get_valid_node_labels()
        features = []
        for l in lbls:
            features.append([x for _,x in sorted(nx.get_node_attributes(self.graph, l).items())])

        #append dummy node
        for f in features:
            f.append(0)

        assert len(features[0]) == len(list(self.graph.nodes))+1, "number of features deviates from number of nodes" 

        return np.array(features).transpose().astype(np.float64)

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
        for o, i, w in self.graph.out_edges(data=True):
            out_edges_dict[o] = out_edges_dict.get(o, list())+ [(i,list(w.values())[0])]
        return self.__convert_dict_to_node_and_weight_list(out_edges_dict)


    def __extract_in_sample(self):
        """
        Extracts the  deterministic sampling of size equal to neighb_size
        for the incoming neighbourhood. The deterministic sampling in based 
        on edge weight in order from high to low

        @return a tuple with the first element a 2d numpy with the incoming 
                adjacent node ids of the sample per node. The second element 
                is a 2d numpy array containing the edge weights of the sample 
                per node 
        """
        in_edges_dict = {}
        for o, i, w in self.graph.in_edges(data=True):
            in_edges_dict[i] = in_edges_dict.get(i, list())+ [(o,list(w.values())[0])]
        return self.__convert_dict_to_node_and_weight_list(in_edges_dict)

    def __convert_dict_to_node_and_weight_list(self, edges_dict):
        """
        Helper function that converts a dictionary with tuples of neighbours
        and weight into a nodes sample and weight sample of size equal to 
        neighb_size based on a sorted weight. if required the neighbouring 
        nodes are extended with dummy nodes up to the neighb_size.
        """
        dummy_id = self.features.shape[0]-1
        nodes = list(self.graph.nodes)
        for k in nodes:
            v = sorted(edges_dict.get(k,[(dummy_id, DataFeederNx.DUMMY_WEIGHT)]), key = lambda x: x[1], reverse=True)
            if len(v) <= self.neighb_size:
                v = v + [(dummy_id, DataFeederNx.DUMMY_WEIGHT)] * (self.neighb_size - len(v))
            else:
                v = v[0:self.neighb_size] 
            edges_dict[k] = v

        edges_list = []
        weight_list = []
        for  _, v in sorted(edges_dict.items()):
            edges_list.append([t[0] for t in v])
            weight_list.append([t[1] for t in v])

        edges_list.append([dummy_id] * self.neighb_size)
        weight_list.append([DataFeederNx.DUMMY_WEIGHT] * self.neighb_size)

        return (np.array(edges_list), np.array(weight_list).astype(np.float64))


    def __get_valid_node_labels(self):
        """
        Checks which node labels are set for all nodes. Labels which are only
        populated for a limited part of the nodes in the graph are excluded as
        labels, all others are included as node features.
        """
        node_label_stats = {}
        for i in self.graph.nodes.keys():
            for lbl in  self.graph.nodes[i].keys():
                node_label_stats[lbl] = node_label_stats.get(lbl, 0) + 1

        max_value = max(node_label_stats.values())
        incl_list = []
        excl_list = []
        for k,v in node_label_stats.items():
            if v == max_value:
                incl_list.append(k)
            else:
                excl_list.append(k)

        self.feature_dim = len(incl_list)
        if self.verbose:
            print(f"The following labels are excluded {excl_list}")
            print(f"The following labels are included {incl_list}")
        return incl_list

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

    def check_graph(self, G):
        '''
        Check if the graph is directed.
        '''
        assert nx.is_directed(G), "Only Directed graph are currently supported"
        

    def get_train_samples(self):
        return self.iter["train"]

    def get_val_samples(self):
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
        return self.feature_dim