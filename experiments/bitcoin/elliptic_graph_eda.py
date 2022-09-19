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
#%%
"""#####################
creates a networkx graph of the elliptics dataset
refer to Elliptic, www.elliptic.co.

steps
1. load nodes in pandas and select appropriate columns
2. normal the feature values between 0 and 1
3. reindex the nodes from 0 
4. load edge info and redefine with the reindex.
5. create graph.
"""
#%% define paths

base_folder = ROOT_FOLDER + '/data/bitcoin/'
node_path = base_folder + 'elliptic_txs_features.csv'
edge_path = base_folder + 'elliptic_txs_edgelist.csv'
label_path = base_folder + 'elliptic_txs_classes.csv'

#%% 1. load nodes in pandas and select appropriate columns

nodes = pd.read_csv(node_path, header=None)
cnt_orig = nodes.shape
labels = pd.read_csv(label_path)
nodes.columns = ['bitcoin_id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
nodes = pd.merge(labels, nodes, left_on='txId', right_on='bitcoin_id', how='right')
nodes.drop(labels='txId', axis=1, inplace=True)
cnt_new = nodes.shape
print(f'rows orig {cnt_orig[0]}, rows news {cnt_new[0]}')

# %% normal the feature values between 0 and 1
df = nodes.iloc[:, 3:]  # first two columns are id and time frame
norm_nodes = (df-df.min())/(df.max()-df.min())
norm_nodes = pd.concat([nodes.iloc[:, :3], norm_nodes], axis=1)
# %% reindex the nodes from 0
norm_nodes['id'] = norm_nodes.index

#%% load edge info and redefine with the reindex.

edges = pd.read_csv(edge_path)
cnt_orig = edges.shape
node_mapping = norm_nodes[['id', 'bitcoin_id']]

# join mapping with the source and target.
edge = pd.merge(edges, node_mapping, left_on='txId1', right_on='bitcoin_id', how='left')
edge.rename(columns={'id':'source'}, inplace=True)
edge.drop(labels='bitcoin_id', axis=1, inplace=True)
edge = pd.merge(edge, node_mapping, left_on='txId2', right_on='bitcoin_id', how='left')
edge.rename(columns={'id':'target'}, inplace=True)
edge.drop(labels='bitcoin_id', axis=1, inplace=True)
edge['weight'] = 1.0

cnt_new = edge.shape
cnt_na = edge.isna().sum()
print(f'rows orig {cnt_orig[0]}, rows news {cnt_new[0]}, count na {cnt_na}')
# %% create graph
G = nx.from_pandas_edgelist(edge, 'source', 'target', ["weight"], create_using=nx.DiGraph)

for col in norm_nodes.columns[:10]:
    if col not in ['bitcoin_id', 'time step', 'id']:
        nx.set_node_attributes(G, pd.Series(norm_nodes[col], index=norm_nodes.id).to_dict(), col)

#%% eda - feature list

# norm_nodes.describe()
# norm_nodes.isna().sum().sum()

#%% EDA - create partial graph
depth = 4
node_id = 8
# sub_G = G.subgraph(nx.node_connected_component(nx.to_undirected(G), node_id)).copy()
local_graph = list(nx.single_source_dijkstra_path_length(G, node_id, depth).keys())
sub_G = G.subgraph(local_graph)

pos = nx.kamada_kawai_layout(sub_G)
# color = [(i % 34) / 34 for i in graph.nodes()]
options = {
    # 'node_color': color,
    'arrows': True,
    'node_size': 100,
    'width': 1,
    'with_labels': False,
    'pos': pos,
    'cmap': plt.cm.rainbow
}
nx.draw(sub_G, **options)
plt.show()
# %% explore the degree distribution

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

plot_degree_dist(G)
#%%
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt

# G = some networkx graph

degrees = [G.degree(n) for n in G.nodes() if G.nodes[n]['class'] == 'unknown']
degree_counts = Counter(degrees)                                                                                                 
x, y = zip(*degree_counts.items())                                                      

plt.figure(1)   

# prep axes                                                                                                                      
plt.xlabel('degree')                                                                                                             
plt.xscale('log')                                                                                                                
plt.xlim(1, max(x))  

plt.ylabel('frequency')                                                                                                          
plt.yscale('log')                                                                                                                
plt.ylim(1, max(y))                                                                                                             
                                                                                                                                     # do plot                                                                                                                        
plt.scatter(x, y, marker='.')                                                                                                    
plt.show()

# %%
