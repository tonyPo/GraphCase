#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 07:26:20 2019

@author: tonpoppe

cd /Users/tonpoppe/.local/lib/python3.5/site-packages/tensorboard
python main.py --logdir=/Volumes/GoogleDrive/My Drive/KULeuven/thesis/test_output/
"""

import time
import numpy as np
import pandas as pd
import math
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import pickle

from model import GraphAutoEncoderModel
from dataFeeder import DataFeeder


class GraphAutoEncoder:
    """
    This class implement the graphCase algorithm. Refer for more details    
    """
    def __init__(self, training_folder,
                 output_file,
                 learning_rate=0.0001,
                 weight_decay=0.0,  # 'weight for l2 loss on embedding matrix.'
                 epochs=4,
                 dropout=0.0,
                 support_size=[2, 2],
                 dims=[32, 32, 32, 32],
                 batch_size=3,
                 print_every=5,
                 max_total_steps=100,
                 get_all_embeddings_flag=True,
                 output_param_file="/dbfs/tmp/ton/graphsagmodel2/tmp/model.ckpt",
                 # output_param_file = "tmp/model.ckpt",
                 output_var_file='tmp/variable_values.pickle',
                 output_hist_file='tmp/hist.pickle',
                 batches_per_file=10,
                 validate_iter=5,
                 embedding_file="embedding",
                 data_feeder=None,
                 feed_is_retail=False
                 ):
        self.dropout = dropout
        self.output_file = output_file
        self.epochs = epochs
        self.print_every = print_every
        self.max_total_steps = max_total_steps
        self.get_all_embeddings_flag = get_all_embeddings_flag
        self.batches_per_file = batches_per_file
        self.validate_iter = validate_iter
        self.learning_rate = learning_rate
        self.embedding_file = embedding_file
        self.output_var_file = output_file + output_var_file
        self.output_hist_file = output_file + output_hist_file
        self.history = {}
        if output_param_file is None:
            self.output_param_file = None
        else:
            self.output_param_file = output_param_file
        self.dims = dims
        
        print("init graphCase")
        #create sampler
        if data_feeder is None:
            print("initialize data feeder")
            self.sampler = DataFeeder(training_folder, batch_size, feed_is_retail)
        else:
            self.sampler=data_feeder
        feature_size=self.sampler.get_feature_size()
        #create model

        self.model = GraphAutoEncoderModel(learning_rate,
                                        weight_decay,
                                        dims,
                                        feature_size,
                                        dropout,
                                        logging=True)

        # set feature file and in and out samples
        features= self.sampler.features[:,1:]
        in_sample = self.sampler.in_sample[:,1:]
        out_sample = self.sampler.out_sample[:,1:]
        in_sample_amnt = self.sampler.in_sample_amnt[:,1:]
        out_sample_amnt = self.sampler.out_sample_amnt[:,1:]
        self.model.set_constant_data( features, in_sample, out_sample, in_sample_amnt, out_sample_amnt)



    def train_layer(self, layer, dim=None , learning_rate=None, act=tf.nn.sigmoid):
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
        self.init_history()
        counter = 0
        for i in self.sampler.get_train_samples():

            try:
                l, _ = getattr(self.model, method)(i[0])

                # validation & print step
                if counter % self.validate_iter == 0:
                    val_counter = 0
                    val_loss = 0
                    for j in self.sampler.get_val_samples():
                        val_l, _ = getattr(self.model, method)(j[0], is_validation_step=True)
                        val_loss += val_l
                        val_counter += 1
                        if val_counter == 10:
                            break

                    val_loss = val_loss / val_counter
                        # Print results
                    print("Iter:", '%04d' % counter,
                          "train_loss=", "{:.5f}".format(l),
                          "val_loss=", "{:.5f}".format(val_loss),
                          "time=", time.strftime('%Y-%m-%d %H:%M:%S'))
                    self.update_history(counter, l, val_loss, time.strftime('%Y-%m-%d %H:%M:%S'))

                counter += 1
                if counter == self.max_total_steps:
                    break

            except tf.errors.OutOfRangeError:
                print("reached end of batch via out of range error")
                break

        return self.history

    def calculate_all_embeddings(self):
        print("calculating all embeddings")
        self.sampler.init_incr_batch()
        self.check_dir(self.output_file[:-1], create=True, clean=False)
        # remove old files
        [f.unlink() for f in Path(self.output_file[:-1]).glob(self.embedding_file + '*.csv')]

        embedding = None
        counter = 0
        file_counter = 0
        for i in self.sampler.get_inc_samples():
            counter += 1
            try:
                e = self.model.calculate_embedding(i[0])
                if embedding is None:
                    embedding = e
                else:
                    embedding = np.vstack([embedding,e])
                if counter % 100 == 0:
                    print("processed ", counter, " batches time: ", datetime.now())

                if counter % self.batches_per_file == 0:
                    file_name = self.output_file + self.embedding_file + str(file_counter) + ".pickle"
                    pickle.dump(embedding, open( file_name, "wb"))
                    # np.savetxt(file_name, embedding, delimiter=',')
                    print("Saved embedding batch in file", file_name, " with nr ", file_counter, " and shape ", np.shape(embedding))
                    embedding = None
                    file_counter += 1

            except tf.errors.OutOfRangeError:
                break

        print("reached end of batch")
        file_name = self.output_file + self.embedding_file + str(file_counter) + ".pickle"
        pickle.dump(embedding, open( file_name, "wb"))
        # np.savetxt(file_name, embedding, delimiter=',')
        print("Saved embedding batch in file", file_name, " with nr ", file_counter, " and shape ", np.shape(embedding))



    def check_dir(self, dir_to_check, create=False, clean=True):
        """
        Check if the directory exists.
        If the create = false and the directory does not exist then an error 
        is raised.
        if the create = True and only the last subdirectory does not exists 
        then it is created.
        If the directory does exists then all csv files are removed.
        """
        p = Path(dir_to_check)
        if not p.is_dir():
            base_dir = p.parent
            if create and base_dir.is_dir():
                # create new path
                print("directory created")
                p.mkdir()
                
            else:
                raise Exception("directory does not exists:" + dir_to_check)
        else:
            if clean:
                # check if directory is empty
                [f.unlink() for f in p.glob('*.csv')]

    def init_history(self):
        self.history = {}
        self.history["i"] = []
        self.history["l"] = []
        self.history["val_l"] = []
        self.history["time"] = []

    def update_history(self, i, l, val_l, curtime):
        self.history["i"].append(i)
        self.history["l"].append(l)
        self.history["val_l"].append(val_l)
        self.history["time"].append(curtime)

    def save_model(self, filename):
        pickle.dump(self.model, open(filename, "wb"))

    def load_model(self, filename):
        self.model = pickle.load(open(filename, "rb"))