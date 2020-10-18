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
#%% check sampled neighbourhood

graph = karate.create_karakte_mirror_network({'weight': 'random'}, {'label0': 1, 'label1': 'random'})
graph = karate.create_karakte_mirror_network({'weight': 'random'}, {'label0': 1, 'label1': 'random'})
for node in graph.nodes(data=True):
    node[1]['label0'] = int(node[0])

gae = GraphAutoEncoder(graph, learning_rate=0.01, support_size=[3, 3], dims=[3, 5, 7, 6, 2],
                       batch_size=12, max_total_steps=10, verbose=True, useBN=True)


def plot_node(graph, node_id):
    local_graph = []
    for neightbor in graph.neighbors(node_id):
        local_graph = local_graph + [n for n in graph.neighbors(neightbor)]
    local_graph = list(set(local_graph))  # make list unique
    subgraph = graph.subgraph(local_graph)

    # plot subgraph
    nt = net.Network(notebook=True, directed=True)
    nt.from_nx(subgraph)
    nt.set_edge_smooth('straightCross')
    length_dict = nx.single_source_dijkstra_path_length(subgraph, node_id, 2, weight=1)
    color_dict = {0: 'red', 1: 'lightblue', 2: 'lightgreen'}
    for node in nt.nodes:
        node["color"] = color_dict[length_dict[node['id']]]
        node['shape'] = 'circle'
    for edge in nt.edges:
       edge['label'] = round(edge['weight'], 2)
    nt.toggle_physics(False)
    return nt


nt = plot_node(graph, 6)
nt.show(ROOT_FOLDER + "/temp/nx.html")

#%%
df, _, pv_graph = gae.get_l1_structure(6, node_label='feat0', get_pyvis=True)
# pv_graph.toggle_physics(False)
pv_graph.show(ROOT_FOLDER + "/temp/nx2.html")
pd_df = pd.DataFrame(data=np.squeeze(df), columns=["edge", 'node_id', "label2"])

# %%

# nt = net.Network(directed=True, notebook=True)
# # populates the nodes and edges data structures
# nt.from_nx(graph)
# nt.set_edge_smooth('dynamic')
# nt.show("nx.html")
# %%
