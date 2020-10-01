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
sys.path.insert(0, ROOT_FOLDER)
sys.path.insert(0, ROOT_FOLDER+"/examples")
from  GAE.graph_case_controller import GraphAutoEncoder
import example_graph_bell as gb

#%% constants
TRAIN = True
MODEL_FILENAME = ROOT_FOLDER+"/data/gae_gb"
RESULTS_FILE = ROOT_FOLDER+"/data/train_gb"

#%% create graph

graph = gb.create_directed_barbell(10, 10)
# correction edge weight for node # 20
ndic = graph.nodes(data='label1')
for u, v, d in graph.edges(data=True):
    if(v > 9) & (v < 21):
        d['weight'] = 1
    else:
        d['weight'] = ndic[u] * ndic[v]

#%% create and train model
gae = GraphAutoEncoder(graph, learning_rate=0.01, support_size=[5, 5], dims=[3, 5, 7, 6, 2],
                       batch_size=30, max_total_steps=1000, verbose=True, act=tf.nn.tanh)
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
color = [outdeg[y]/10 for y in range(node_count)]
edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
options = {
    'node_color': color,
    'node_size': 300,
    'edgelist':edges,
    'edge_color':weights,
    'width': 1,
    'with_labels': True,
    'edge_cmap': plt.cm.Paired,
    'cmap': plt.cm.gist_rainbow,
    'pos': pos
}
nx.draw(graph, **options)
plt.show()

#%% plot embeddings

cm_col = plt.cm.get_cmap('gist_rainbow', node_count)
colormp = [cm_col(x) for x in color]
plt.scatter(embed[:node_count, 1], embed[:node_count, 2], c=colormp, label='embedding')

#%% plot training results
train_res = pickle.load(open(MODEL_FILENAME, "rb"))
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

# %%
