#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:22:45 2019

@author: tonpoppe
"""

# MAC OS bug
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import datetime
from multiprocessing.pool import ThreadPool

# Helper libraries
import numpy as np
from pathlib import Path
import os
import csv

class DataFeeder:
    """
    This class reads the samples file in CSV format and feeds the samples to the
    graphSage training algorithm.
    For training it needs a  sample set for batch 1, batch 2 and negative
    examples batch.
    """

    def __init__(self, import_dir, batch_size=3):
        self.import_dir = import_dir
        self.batch_size = batch_size
        # self.support_sizes = self.get_sizes(support_size)
        self.iter = {}
 
        self.features = self.load_files("node_labels")
        self.in_sample = self.load_files("in_sample" , np.int32)
        self.out_sample = self.load_files("out_sample", np.int32)
        self.in_sample_amnt = self.load_files("in_sample_amnt")
        self.out_sample_amnt = self.load_files("out_sample_amnt")
        self.feature_dim = np.shape(self.features)[1]-1

    def init_train_batch(self):
        if "train" not in  self.iter:
            self.iter["train"] = self.create_sample_iterators("train", self.batch_size)
            self.iter["valid"] = self.create_sample_iterators("valid", self.batch_size)

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

    def create_sample_iterators(self, dataset_type, size):
        """
        Creates an dataset iterator
        :param dataset_type: {t_s, v_s}
        :param size:
        :param repeat:
        :return:
        """
        print("iterator for dataset " + dataset_type)
        file_name = self.get_fs(dataset_type, 'csv')
            # first column is the id in string format
        print(file_name)
        record_defaults = [tf.int64]
        dataset = tf.data.experimental.CsvDataset(file_name,
                                                  record_defaults,
                                                  # compression_type = "GZIP",
                                                  # buffer_size=1024*1024*1024,
                                                  header=True)
        dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        batched_dataset = dataset.batch(size, drop_remainder=True).repeat(count=-1)
        return batched_dataset

    # def batch_to_iterator(self, batchedDataset):
    #     iter = tf.data.Iterator.from_structure(batchedDataset.output_types,
    #                                            batchedDataset.output_shapes)
    #     iter.make_initializer(batchedDataset)
    #     return iter

    def get_f(self, filename):
        """
        retrieves the first partition of the filename in the hadoop environment.
        Hadoop stored the data in a subfolder of the filename which is actually
        a folder.
        """
        f_base = Path(self.import_dir + filename)
        #query al csv files in the folder and return the first file
        f_name = list(f_base.glob('*.csv'))
        return str(f_name[0])

    def get_fs(self, filename, file_ext = 'gz'):
        """
        retrieves the first partition of the filename in the hadoop environment.
        Hadoop stored the data in a subfolder of the filename which is actually
        a folder.
        """
        f_base = Path(self.import_dir + filename)
        #query al csv files in the folder and return the first file
        f_name = list(f_base.glob('*.'+file_ext))
        #order list alphabetically
        #        f_name_str = [str(f_name[x]) for x in range(len(f_name))]
        f_name_str = []
        for i in range(len(f_name)):
            f_name_str.append(str(f_name[i]))
        f_name_str.sort()
        return f_name_str

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