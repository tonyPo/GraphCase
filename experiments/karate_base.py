#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This notebook create the mirrored karate network and trains a basic graphcase on this network.
It shows the resulting embedding in a scatter plot.
'''
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
TRAIN = True
MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar_batch2"
RESULTS_FILE = ROOT_FOLDER+"/data/train_kar_batch2"
#%% --- build karate network

G = nx.read_edgelist(KARATE_FILE)
G = G.to_directed()

# assign random edge labels
random.seed(1)
for u, v, d in G.edges(data=True):
    d['weight'] = random.uniform(0.3, 1.0)

# assign random node labels
for u in G.nodes(data=True):
    u[1]['label1'] = 0.5  #random.uniform(0.0, 1.0)
    u[1]['label2'] = 0.5 #random.uniform(0.0, 1.0)

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
color = [(i % 34) / 34 for i in graph.nodes()]
options = {
    'node_color': color,
    'arrows': False,
    'node_size': 200,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'cmap': plt.cm.rainbow
}
nx.draw(graph, **options)
plt.show()
#%% train GAE en calculate embeddings

gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[1, 6, 6, 6],
                       hub0_feature_with_neighb_dim=2, batch_size=16, verbose=True,
                       act=tf.nn.relu, seed=1, dropout=0.1)

if TRAIN:
    history = gae.fit(epochs=100, layer_wise=False)
    plt.plot(history[None].history['loss'], label='loss')
    plt.plot(history[None].history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
 
    pickle.dump(history[None].history, open(RESULTS_FILE, "wb"))
    gae.save_weights(MODEL_FILENAME)
else:
    gae.load_weights(MODEL_FILENAME)

embed = gae.calculate_embeddings()


# %%
indeg = graph.in_degree()
outdeg = graph.out_degree()
tbl = np.array([[y, x['label1'], indeg[y], outdeg[y], embed[y, 1], embed[y, 2]]
                for y, x in graph.nodes(data=True)])
pd_tbl = pd.DataFrame(tbl[:, 1:], tbl[:, 0],
                      ['label1', 'in_degree', 'out_degree', 'embed1', 'embed2'])
print(pd_tbl)
# %%
node_count = graph.number_of_nodes()
import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=embed[:node_count, 1],
                                y=embed[:node_count, 2],
                                mode='markers',
                                marker=dict(
                                    color=pd_tbl.index % 34 / 34,
                                    colorscale = 'rainbow'
                                ),
                                text=embed[:node_count, 0])) # hover text goes here

fig.update_layout(title='embedding values of karate network')
fig.show()
# %% train loss

train_res = pickle.load(open(RESULTS_FILE, "rb"))
plt.plot(train_res['loss'], label='train')
plt.plot(train_res['val_loss'], label='validation')
plt.legend()
plt.yscale('log')
plt.xlabel("iteration")
plt.ylabel("validaiton loss")
plt.show()
train_res['val_loss'][-1]

#%% plot local graphs of node to investage why nodes are close to each other

"""
Both nodes 17 and 20 have a similar embedding but are located
in vary different parts of the network.
"""
from GAE.graph_case_tools import Tools
sys.path.insert(0, ROOT_FOLDER + "/temp/nx2.html")
pnet = Tools.plot_node(graph, 20)
pnet.show(ROOT_FOLDER + "/temp/nx2.html")


# %%  Result of various runs
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