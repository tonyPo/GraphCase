#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:22:45 2019

@author: tonpoppe
"""

# MAC OS bug
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import networkx as nx
import datetime
from multiprocessing.pool import ThreadPool

# Helper libraries
import numpy as np
from pathlib import Path
import os
import csv

class DataFeederNx:
    """
    This class reads a directed network object and covert this to 
    the samples for training the GraphCase algorithm.
    """

    def __init__(self, graph, batch_size=3, val_fraction=0.3):
        #TODO check if graph is bi-directed
        self.val_frac = val_fraction  # fraction of nodes used for validation set
        self.batch_size = batch_size  # number of nodes in a training batch
        self.graph = graph
        self.iter = {}
        self.features = self.__extract_features()
        # self.in_sample = self.load_files("in_sample" , np.int32)
        # self.out_sample = self.load_files("out_sample", np.int32)
        # self.in_sample_amnt = self.load_files("in_sample_amnt")
        # self.out_sample_amnt = self.load_files("out_sample_amnt")
        # self.feature_dim = np.shape(self.features)[1]-1

    def init_train_batch(self):
        """
        Creates a training batch and validation batch based on a random split of the nodes
        in the graph. The batches contain a list of node id's only and are assigned to 
        the iter attribute.
        """
        # split nodes in train set
        train, val = self.__train_test_split()
        self.iter["train"] = self.create_sample_iterators(train, self.batch_size)
        self.iter["valid"] = self.create_sample_iterators(val, self.batch_size)

    def __train_test_split(self):
        nodes = list(self.graph.nodes)
        random.shuffle(nodes)
        split_value = round(self.val_frac * 100)
        train_data = nodes[:split_value]
        test_data = nodes[split_value:]
        return train_data, test_data

    def __extract_features(self):
        lbls = self.__get_valid_node_labels()
        features = []
        for l in lbls:
            features.append([x for _,x in sorted(nx.get_node_attributes(self.graph, l).items())])

        return np.array(features)


    def __get_valid_node_labels(self):
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
        print(f"The following labels are excluded {excl_list}")
        print(f"The following labels are included {incl_list}")
        return incl_list

    def init_incr_batch(self):
        if "incr" not in  self.iter:
            self.iter["incr"] = self.create_incremental_set_iterator(self.batch_size)

    def get_train_samples(self):
        return self.iter["train"]

    def get_val_samples(self):
        return self.iter["valid"]

    def get_inc_samples(self):
        return self.iter["incr"]

    def create_incremental_set_iterator(self, size):
        """
        Creates an dataset iterator
        :param dataset_type: {t_s, v_s}
        :param size:
        :param repeat:
        :return:
        """
        file_name = self.get_fs("incr", 'csv')
        # first column is the id in string format
        record_defaults = [tf.int64]

        dataset = tf.data.experimental.CsvDataset(file_name,
                                                  record_defaults,
                                                  # compression_type = "GZIP",
                                                  # buffer_size=1024*1024*1024,
                                                  header=True)
        dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        batched_dataset = dataset.batch(size, drop_remainder=True)
        return batched_dataset

    def create_sample_iterators(self, node_list, size):
        """
        Converts a list of nodes into an tensorflow dataset for training.

        :param node_list: list of nodes
        :param size: The size of the batches.
        :return: tensorflow  dataset of node list of size size
        """
        record_defaults = [tf.int64]
        dataset = tf.data.Dataset.from_tensor_slices(node_list)
        batched_dataset = dataset.batch(size, drop_remainder=True).repeat(count=-1)
        return batched_dataset



    def load_files(self, filename, data_type= np.float32):
        def load_part_files (f, data_type):
            df = np.loadtxt(f, skiprows=1, delimiter=",", dtype=data_type)
            return df

        print("loading ", filename, " ", datetime.datetime.now())
        file_list = self.get_fs(filename)
        df = None

        pool = ThreadPool(processes=len(file_list))
        async_result = []

        for f in file_list:
            async_result.append(pool.apply_async(load_part_files, (str(f), data_type)  ))

        for i, f in enumerate(file_list):
            return_val = async_result[i].get()
            if np.shape(return_val)[0] > 0:
                if df is None:
                    df = return_val
                else:
                    df = np.vstack((df,return_val))

        # sort the df by the first column
        print("sorting ", filename, " : ", datetime.datetime.now())
        df = df[df[:,0].argsort()]
        # df.astype(data_type)
        print("loading ", filename, " finished ", datetime.datetime.now())
        return df



    def get_feature_size(self):
        return self.feature_dim


#
#if __name__ == '__main__':
#    print("main method")
#    folder = '/Volumes/GoogleDrive/My Drive/KULeuven/thesis/test_output/sampler/out/'
#    feeder = DataFeeder(folder)