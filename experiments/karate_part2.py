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
from pyvis import network as net

ROOT_FOLDER = os.path.dirname(os.getcwd())
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
sys.path.insert(0, ROOT_FOLDER + '/examples/')
sys.path.insert(0, ROOT_FOLDER)

from GAE.graph_case_controller import GraphAutoEncoder
import karate
#%% constant declaration
TRAIN = True
MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar_part2"
RESULTS_FILE = ROOT_FOLDER+"/data/train_kar_part2"

#%% create graph and train embedding
graph = karate.create_karakte_mirror_network({'weight': 'random'}, {'label0': 'random', 'label1': 'random'})

gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[2, 8, 8, 8, 8],
                       batch_size=1024, max_total_steps=10000, verbose=True, act=tf.nn.tanh)
if TRAIN:
    train_res = {}
    for i in range(len(gae.dims)):
        if i in [1, 2]:
            train_res["l"+str(i+1)] = gae.train_layer(i+1, dropout=0.1)
        else:
            train_res["l"+str(i+1)] = gae.train_layer(i+1)

    train_res['all'] = gae.train_layer(len(gae.dims), all_layers=True)
    pickle.dump(train_res, open(RESULTS_FILE, "wb"))
    gae.save_model(MODEL_FILENAME)
else:
    gae.load_model(MODEL_FILENAME, graph)

embed = gae.calculate_embeddings()

#%% calculate distance between corresponding and all pairs
size = 34

delta_cor = sum([np.subtract(embed[i], embed[i+size]) for i in range(size)])
delta_cor = delta_cor / size

delta_all = 0
for i in range(size):
    delta_all = delta_all + sum([np.subtract(embed[i], embed[i + size + j]) for j in range(size - i)])
delta_all = delta_all / ((size -1) * size / 2)