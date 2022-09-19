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
ROOT_FOLDER = os.path.dirname(os.getcwd())
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
os.chdir(ROOT_FOLDER)
sys.path.insert(0, ROOT_FOLDER)
sys.path.insert(0, ROOT_FOLDER+"/examples")
from GAE.graph_case_controller import GraphAutoEncoder
import example_graph_bell as gb

#%% constants
TRAIN = True
MODEL_FILENAME = ROOT_FOLDER+"/data/gae_gb"
RESULTS_FILE = ROOT_FOLDER+"/data/train_gb"

#%% create graph

graph = gb.create_directed_barbell(10, 10)
# connect second buble to the same node as in the first
graph.remove_edge(21, 20)
graph.add_edge(29,20, weight=1)

# correction edge weight for node # 20
ndic = graph.nodes(data='label1')
for u, v, d in graph.edges(data=True):
    if(v > 9) & (v < 21):
        d['weight'] = 1
    else:
        d['weight'] = ndic[u] * ndic[v]

graph = gb.create_directed_barbell2(10, 10)
#%% create and train model
gae = GraphAutoEncoder(
    graph, support_size=[5, 5], dims=[3,5,7,6], batch_size=8, hub0_feature_with_neighb_dim=2,
    useBN=True, verbose=False, seed=1, learning_rate=0.01, act=tf.nn.sigmoid)

# gae = GraphAutoEncoder(graph, learning_rate=0.01, support_size=[5, 5], dims=[3, 5, 7, 6, 2],
#                        batch_size=30, max_total_steps=1000, verbose=True, act=tf.nn.tanh)
if TRAIN:
    history = gae.fit(epochs=1000, layer_wise=False)
    pickle.dump(history, open(RESULTS_FILE, "wb"))
    gae.save_weights(MODEL_FILENAME)
else:
    gae.load_weights(MODEL_FILENAME)

embed = gae.calculate_embeddings()

# %% get tabel with node details
indeg = graph.in_degree()
outdeg = graph.out_degree()
tbl = np.array([[y, x['label1'], x['label2'], indeg[y], outdeg[y], embed[y, 1], embed[y, 2]]
                for y, x in graph.nodes(data=True)])
pd_tbl = pd.DataFrame(tbl[:, 1:], tbl[:, 0],
                      ['label1', 'label2', 'in_degree', 'out_degree', 'embed1', 'embed2'])
print(pd_tbl)

#%%  show graph
pos = nx.kamada_kawai_layout(graph, scale=10, weight=None)
node_count = graph.number_of_nodes()
outdeg = graph.out_degree()
colorcodes = {'yellow': 0.1, 'orange': 0.2, 'green': 0.3, 'lightblue': 0.6, 'darkblue': 0.9}
yellow = [0, 2, 4, 6, 8, 22, 24, 26, 28, 29]
orange = [1, 3, 5, 7, 21, 23, 25, 27]
green = [9, 30]
lightblue = [15]
darkblue = [10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
color = [(colorcodes[y], x) for y in ["yellow", 'orange', 'green', 'lightblue', 'darkblue'] for x in globals()[y]]
color.sort(key = lambda x: x[1])
color = [y for (y,x) in color]
cm_col = plt.cm.get_cmap('gist_rainbow', 1000)
colormp = [cm_col(x) for x in color]
print(color)
# color = [outdeg[y]/10 for y in range(node_count)]
edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
options = {
    'node_color': colormp,
    'node_size': 300,
    'edgelist':edges,
    'edge_color':weights,
    'width': 1,
    'with_labels': True,
    'edge_cmap': plt.cm.Paired,
    # 'cmap': plt.cm.gist_rainbow,
    'pos': pos
}
nx.draw(graph, **options)
plt.show()

#%% plot embeddings

cm_col = plt.cm.get_cmap('gist_rainbow', 1000)
colormp = [cm_col(x) for x in color]
plt.scatter(embed[:node_count, 1], embed[:node_count, 2], c=colormp, label='embedding')

#%% plot training results
# history = pickle.load(open(RESULTS_FILE, "rb"))
plt.plot(history[None].history['loss'], label='loss')
plt.plot(history[None].history['val_loss'], label='val_loss')
plt.legend()
plt.yscale('log')
plt.xlabel("iteration")
plt.ylabel("validaiton loss")
plt.show()

# %%
