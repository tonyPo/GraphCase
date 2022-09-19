#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import random

ROOT_FOLDER = os.path.dirname(os.getcwd())
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
sys.path.insert(0, ROOT_FOLDER)
sys.path.insert(0, ROOT_FOLDER + '/experiments/bitcoin')
import bitcoin_graph as bg
from  GAE.graph_case_controller import GraphAutoEncoder

#%% description
'''
Apply a gridsearch on the bitcoin graph
the get the optimal hyper params for a 32 bit embed
'''

#%% parameter

base_folder = ROOT_FOLDER + '/data/bitcoin/'
learning_rates = [0.001, 0.005, 0.0005]
dropout_levels = [0, 1, 2, 3]
act_functions = [tf.nn.tanh, tf.nn.sigmoid]
dim_size = 32
epochs = 40
save_folder = ROOT_FOLDER + '/data/bitcoin/models/'

#%% grid search

graph = bg.create_bitcoin_graph(base_folder)
gs_res = {}

for lr in learning_rates:
    for do in dropout_levels:
        for act in act_functions:
            dims = [dim_size]*5
            gae = GraphAutoEncoder(graph, learning_rate=lr, support_size=[5, 5], dims=dims,
                                   batch_size=1024, max_total_steps=epochs, verbose=True, act=act)

            train_res = {}
            for i in range(len(gae.dims)):
                if i in range(1, 1 + do):
                    train_res["l"+str(i+1)] = gae.train_layer(i+1, dropout=0.1)
                else:
                    train_res["l"+str(i+1)] = gae.train_layer(i+1, dropout=None)

            train_res['all'] = gae.train_layer(len(gae.dims), all_layers=True)
            
            # save results
            act_str =  'tanh' if act==tf.nn.tanh else 'sigm'
            run_id = f'dim_{dim_size}_lr_{lr}_do_{do}_act_{act_str}'
            pickle.dump(train_res, open(save_folder + 'res_' + run_id, "wb"))
            gae.save_model(save_folder + 'mdl_' + run_id)

            # print and store result
            val_los = sum(train_res['all']['val_l'][-4:]) / 4
            gs_res[run_id] = val_los
            print(f'dims:{dim_size}, lr:{lr}, dropout lvl:{do}, act func:{act_str} resultsing val loss {val_los}')

for k, v in gs_res.items():
    print(f'run: {k} with result {v}')

