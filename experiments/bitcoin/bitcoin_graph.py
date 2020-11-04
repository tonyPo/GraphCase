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

def load_and_prep_nodes(base_folder):
    '''
    loads nodes, applies min max normalisation and resets the index 
    '''
    # files
    node_path = base_folder + 'elliptic_txs_features.csv'
    label_path = base_folder + 'elliptic_txs_classes.csv'

    # nodes
    nodes = pd.read_csv(node_path, header=None)
    nodes.columns = ['bitcoin_id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
    labels = pd.read_csv(label_path)

    # merge with labels
    nodes = pd.merge(labels, nodes, left_on='txId', right_on='bitcoin_id', how='right')
    nodes.drop(labels='txId', axis=1, inplace=True)

    return nodes

def load_and_prep_norm_nodes(base_folder):
    nodes = load_and_prep_nodes(base_folder)

    # min-max normalize values
    df = nodes.iloc[:, 3:]  # first two columns are id and time frame
    norm_nodes = (df-df.min())/(df.max()-df.min())
    norm_nodes = pd.concat([nodes.iloc[:, :3], norm_nodes], axis=1)

    # reset id starting from 0
    norm_nodes['id'] = norm_nodes.index

    return norm_nodes

def load_and_prep_edges(base_folder, nodes):
    '''
    loads 
    '''
    # load edges
    edge_path = base_folder + 'elliptic_txs_edgelist.csv'
    edges = pd.read_csv(edge_path)

    # create mapping table
    node_mapping = nodes[['id', 'bitcoin_id']]

    # join mapping with the source.
    edges = pd.merge(edges, node_mapping, left_on='txId1', right_on='bitcoin_id', how='left')
    edges.rename(columns={'id':'source'}, inplace=True)
    edges.drop(labels='bitcoin_id', axis=1, inplace=True)

    edges = pd.merge(edges, node_mapping, left_on='txId2', right_on='bitcoin_id', how='left')
    edges.rename(columns={'id':'target'}, inplace=True)
    edges.drop(labels='bitcoin_id', axis=1, inplace=True)

    # set weight attribute
    edges['weight'] = 1.0

    return edges

def create_bitcoin_graph(base_folder):
    '''
    creates a networkx graph of the elliptics dataset
    refer to Elliptic, www.elliptic.co.
    '''
    nodes = load_and_prep_norm_nodes(base_folder)
    edges = load_and_prep_edges(base_folder, nodes)

    G = nx.from_pandas_edgelist(edges, 'source', 'target', ["weight"], create_using=nx.DiGraph)

    for col in nodes.columns:
        if col not in ['bitcoin_id', 'time step', 'id', 'class']:
            nx.set_node_attributes(G, pd.Series(nodes[col], index=nodes.id).to_dict(), col)

    return G

#%% test
# base_folder = ROOT_FOLDER + '/data/bitcoin/'
# G = create_bitcoin_graph(base_folder)

# %%
