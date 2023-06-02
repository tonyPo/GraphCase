#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:22:45 2019

@author: tonpoppe
"""
import random
import tensorflow as tf
from tensorflow import keras
import networkx as nx
import datetime
import numpy as np
from pathlib import Path
import csv
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F

# if 'spark' not in (locals() or globals()):
#     conf = SparkConf().setAppName('appName').setMaster('local')
#     sc = SparkContext(conf=conf)
#     spark = SparkSession(sc)s
class DataFeederGraphFrames:
    """
    This class reads a node and edge dataframe and covert this to
    the samples for training the GraphCase algorithm.
    """
    def __init__(
        self, G, neighb_size=3, batch_size=3, val_fraction=0.3, verbose=False,
        seed=1, weight_label='weight', encoder_labels=None, label_name=None):
        #TODO set custom dummy node
        #TODO check if ID is ordered starting from zero:
        self.val_frac = val_fraction  # fraction of nodes used for validation set
        self.batch_size = batch_size  # number of nodes in a training batch
        self.neighb_size = neighb_size  # size of the neighborhood sampling
        self.edge_df = G[1]
        self.node_df = G[0]
        self.verbose = verbose
        self.weight_label = weight_label
        self.node_labels = encoder_labels  # labels used for training the encoder
        self.supervised_label = label_name  #label used for supervised training
        self.iter = {}
        self.feature_dim = 0
        self.lbls=None  # vector containing the node labels for supervised training
        self.dummy_weigth = 0
        self.dummy_id = None
        self.features = self.__extract_node_features()
        self.edge_labels = self.__get_valid_edge_labels()
        self.in_sample, self.in_sample_weight = self.__extract_sample(direction='in')
        self.out_sample, self.out_sample_weight = self.__extract_sample(direction='out')
        self.seed = seed
        self.train_epoch_size = 0
        self.val_epoch_size = 0
        

    def init_train_batch(self, **kwargs):
        """
        Creates a training batch and validation batch based on a random split of the nodes
        in the graph. The batches contain a list of node id's only and are assigned to
        the iter attribute.
        """
        # split nodes in train set
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
        nodes = np.squeeze(np.array(self.node_df.select('id').dropDuplicates().collect()))
        random.Random(self.seed).shuffle(nodes)
        split_value = round(self.val_frac * len(nodes))
        test_data = nodes[:split_value]
        train_data = nodes[split_value:]
        return train_data, test_data

    def __extract_node_features(self):
        """
        Creates a feature matrix based on the node df.
        Only columns listed in the node_labels attribute are extracted.

        returns:    2d numpy array of features.
        """
        lbls = self.__get_valid_node_attributes()
        features = self.node_df.select(lbls + ['id'])
        features = features.orderBy('id', ascending=True).drop('id')
        features = np.array(features.collect()).astype(np.float32)

        #append dummy node
        dummy = np.zeros(len(lbls), dtype=np.float32)
        
        return np.vstack([features, dummy]).astype(np.float32)

    def __extract_sample(self, direction='in'):
        """
        Extracts the  deterministic sampling of size equal to neighb_size
        for the incoming neighbourhood. The deterministic sampling in based
        on edge weight in order from high to low

        @return a tuple with the first element a 2d numpy with the incoming
                adjacent node ids of the sample per node. The second element
                is a 3d numpy array containing the edge weights of the sample
                per node.
        """
        if direction == 'in':
            party = 'dst'
            counterparty = 'src'
        else:
            party = 'src'
            counterparty = 'dst'

        edge = (
            self.node_df.select('id')
            .dropDuplicates()
            .join(self.edge_df, self.node_df.id==self.edge_df[party], 'left')
            .drop(party)
        )

        # determine node number of the dummy node
        if self.dummy_id is None:
            self.dummy_id = self.node_df.count()  # count starts with zero

        edge = (edge
            .withColumn(
                'edge_features',
                F.when(F.col(counterparty).isNull(), None)
                    .otherwise(F.array(self.edge_labels))
                )
            )

        w = Window.partitionBy('id').orderBy(-F.col('weight')).rowsBetween(0, self.neighb_size-1)
        w2 = Window.partitionBy('id').orderBy(-F.col('weight'))

        agg_edges = (edge
            .withColumn('sample', F.collect_list(counterparty).over(w))
            .withColumn('features', F.collect_list('edge_features').over(w))
            .withColumn('rn', F.row_number().over(w2))
            .filter("rn = 1")
            .select('id', 'sample', 'features')
            .withColumn('size', F.size(F.col('sample')))
            .withColumn('dummy_node', F.array_repeat(F.lit(self.dummy_id), self.neighb_size - F.col('size')))
            .withColumn('dummy_edge_features',  
                        F.array_repeat(
                            F.array_repeat(F.lit(0),
                                len(self.edge_labels)),
                            self.neighb_size - F.col('size'))
                        )
            .withColumn('sample', F.concat(F.col('sample'), F.col('dummy_node')))
            .withColumn('features', F.concat(F.col('features'), F.col('dummy_edge_features')))
            .orderBy('id')
        )
        sample = np.array(agg_edges.select('sample').collect())
        features = np.array(agg_edges.select('features').collect(), dtype=np.float32)

        #add dummy id
        sample = np.vstack([np.squeeze(sample, 1), [self.dummy_id] * self.neighb_size])
        features = np.vstack(
            [np.squeeze(features, 1),
             np.array([[[0] * len(self.edge_labels)] * self.neighb_size])
            ])

        return (sample, features.astype(np.float32))

    def __get_valid_edge_labels(self):
        """
        Retrieves a list of valid edge labels.
        All labels in the edge dataframe are assumed to be valid.

        Returns:
            list of valid edge labels.
        """
        incl_list = self.edge_df.columns
        incl_list.remove('src')
        incl_list.remove('dst')

        # set edge weight as first item in the list
        try:
            incl_list.remove(self.weight_label)
        except ValueError:
            print(f"ERROR {self.weight_label} doesn't seem to be a valid label")
            exit()
        incl_list = [self.weight_label] + incl_list

        if self.verbose:
            print(f"The following edge labels are included {incl_list}")
        return incl_list

    def __get_valid_node_attributes(self):
        """
        Retrieve the node attributes based on the node_labels attribute.
        If this is not set, then all labels are used.
        If the label name for supervised learning is set,
        then this label is also removed from the node attributes.

        @return a list of node label that are included in the embedding
        """
        cols = self.node_df.columns
        cols.remove('id')
        if self.node_labels:
            # the encoder labels are explicitely set
            valid_attributes = [c for c in cols if c in self.node_labels]
            not_found = [c for c in self.node_labels if c not in cols]
            assert len(not_found) == 0, \
                f"The follwing encoders labels could not be found {not_found}"
        else:
            valid_attributes = cols


        if self.supervised_label:
            # a label is set for supervised learning
            valid_attributes = [c for c in valid_attributes if c!=self.label_name]

        self.feature_dim = len(valid_attributes)
        if self.verbose:
            print(f"The following node labels are included {valid_attributes}")
        return valid_attributes

    def init_incr_batch(self, nodes=None):
        """
        Creates an dataset iterator containing all nodes in the dataset
        @param nodes a list of nodes that need to be included in the dataset
                    if not populated then all nodes of the graph are taken.

        @return tf batches dataset
        """
        if not nodes:
            nodes = np.squeeze(np.array(self.node_df.select('id').dropDuplicates().collect()))

        #amend list with dummy nodes to make len is mulitple of the size.
        if len(nodes) % self.batch_size != 0:
            nodes = np.hstack([
                nodes, [self.dummy_id] * (self.batch_size - (len(nodes) % self.batch_size))
            ])

        dataset = tf.data.Dataset.from_tensor_slices(nodes)
        batched_dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return batched_dataset

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
        """ returns the number of node features used
        """
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

