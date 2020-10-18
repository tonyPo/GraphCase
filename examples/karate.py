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


def create_karakte_network(edge_labels, node_labels):
    """
    creates the karakte network as a networkx graph
    args:
        edge_weight : A dict with values label name : {"random" | float} value used for the edge
                    weight. If random is selected then a uniform distributed random value is used.
        node_labels : A dict with the node label name and value.

    returns:
        A networkx graph of the karakte club
    """

    KARATE_FILE = ROOT_FOLDER + '/data/karate_edges_77.txt'

    # build karate network
    G = nx.read_edgelist(KARATE_FILE)
    G = G.to_directed()
    # convert id to int
    mapping = dict([(x, int(x) - 1) for x in list(G.nodes)])
    G = nx.relabel_nodes(G, mapping)

    # assign edge labels
    random.seed(1)
    for _, _, edge in G.edges(data=True):
        for key, val in edge_labels.items():
            if val == 'random':
                edge[key] = random.uniform(0.3, 1.0)
            else:
                edge[key] = val

    # assign random node labels
    for u in G.nodes(data=True):
        for key, val in node_labels.items():
            if val == 'random':
                u[1][key] = random.uniform(0.0, 1.0)
            else:
                u[1][key] = val

    return G


def create_karakte_mirror_network(edge_labels, node_labels):
    """
    creates the mirrored karakte network as a networkx graph
    args:
        edge_weight : A dict with values label name : {"random" | float} value used for the edge
                    weight. If random is selected then a uniform distributed random value is used.
        node_labels : A dict with the node label name and value.

    returns:
        A networkx graph of the karakte club
    """
    graph = create_karakte_network(edge_labels, node_labels)

    # add the mirrored part of the network
    graph_mirror = graph.copy()
    offset = max(list(graph.nodes)) + 1
    mapping = dict([(x, x + offset) for x in list(graph_mirror.nodes)])
    graph_mirror = nx.relabel_nodes(graph_mirror, mapping)
    graph = nx.union_all([graph, graph_mirror])

    #add single connection
    graph.add_edge(1, 1 + offset, weight=0.5)
    graph.add_edge(1 + offset, 1, weight=0.5)

    return graph

# G = create_karakte_network({'weight': 1.0}, {'label0': 1, 'label1': 1})
# G_D = create_karakte_mirror_network({'weight': 1.0}, {'label0': 1, 'label1': 1})
# nx.draw(G_D)

# %%
