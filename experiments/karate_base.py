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
from  GAE.graph_case_controller import GraphAutoEncoder

#%% -- constant declaration
KARATE_FILE = ROOT_FOLDER + '/data/karate_edges_77.txt'
TRAIN = False
MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar_batch"
RESULTS_FILE = ROOT_FOLDER+"/data/train_kar_batch"
#%% --- build karate network

G = nx.read_edgelist(KARATE_FILE)
G = G.to_directed()

# assign random edge labels
random.seed(1)
for u, v, d in G.edges(data=True):
    d['weight'] = 1.0  # random.uniform(0.3, 1.0)

# assign random node labels
for u in G.nodes(data=True):
    u[1]['label1'] = 1.0  # random.uniform(0.0, 1.0)
    u[1]['label2'] = 1.0  # random.uniform(0.0, 1.0)

# convert id to int
mapping = dict([(x, int(x) - 1) for x in list(G.nodes)])
G = nx.relabel_nodes(G, mapping)
#%%
# add the mirrored part of the network
G_mirror = G.copy()
offset = max(list(G.nodes)) + 1
mapping = dict([(x, x + offset) for x in list(G_mirror.nodes)])
G_mirror = nx.relabel_nodes(G_mirror, mapping)
graph = nx.union_all([G, G_mirror])

#add single connection
graph.add_edge(1, 1 + offset, weight=0.5)
graph.add_edge(1 + offset, 1, weight=0.5)
#%% plot mirrored karate network

pos = nx.kamada_kawai_layout(graph)
options = {
    # 'node_color': color,
    'node_size': 100,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'cmap': plt.cm.Dark2
}
nx.draw(graph, **options)
plt.show()
#%% train GAE en calculate embeddings

gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[3, 8, 8, 6, 2],
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


# %%
indeg = graph.in_degree()
outdeg = graph.out_degree()
tbl = np.array([[y, x['label1'], x['label2'], indeg[y], outdeg[y], embed[y, 1], embed[y, 2]]
                for y, x in graph.nodes(data=True)])
pd_tbl = pd.DataFrame(tbl[:, 1:], tbl[:, 0],
                      ['label1', 'label2', 'in_degree', 'out_degree', 'embed1', 'embed2'])
print(pd_tbl)
# %%
node_count = graph.number_of_nodes()
cm_col = plt.cm.get_cmap('gist_rainbow', 1000)
colormp = [cm_col(x) for x in pd_tbl.index]
plt.scatter(embed[:node_count, 1], embed[:node_count, 2], c=colormp, label='embedding')

# %% train loss

train_res = pickle.load(open(RESULTS_FILE, "rb"))
plt.plot(train_res['all']['i'], train_res['all']['val_l'], label='all')
plt.plot(train_res['l1']['i'], train_res['l1']['val_l'], label='l1')
plt.plot(train_res['l2']['i'], train_res['l2']['val_l'], label='l2')
plt.plot(train_res['l3']['i'], train_res['l3']['val_l'], label='l3')
plt.plot(train_res['l4']['i'], train_res['l4']['val_l'], label='l4')
plt.plot(train_res['l5']['i'], train_res['l5']['val_l'], label='l5')
plt.legend()
plt.yscale('log')
plt.xlabel("iteration")
plt.ylabel("validaiton loss")
plt.show()
train_res['all']['val_l'][-1]
# %%
# 7.5958901166915895
# gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[3, 8, 8, 6, 2],
#                        batch_size=1, max_total_steps=10000, verbose=True, act=tf.nn.tanh)
# MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar"
# RESULTS_FILE = ROOT_FOLDER+"/data/train_kar"

# 20.627742195129393
# gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[3, 8, 8, 6, 2],
#                        batch_size=1, max_total_steps=10000, verbose=True, act=tf.nn.relu)
# MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar_relu"
# RESULTS_FILE = ROOT_FOLDER+"/data/train_kar_relu"


# 6.7972869873046875
# 6960.421875
# gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[3, 8, 8, 6, 2],
#                        batch_size=1024, max_total_steps=10000, verbose=True, act=tf.nn.tanh)
# MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar_batch"
# RESULTS_FILE = ROOT_FOLDER+"/data/train_kar_batch"

# 7032.3640625 for signoid

# BN2 15571.608203125
# BN2 LR 0.01