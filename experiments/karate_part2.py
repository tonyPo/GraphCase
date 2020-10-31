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
from sklearn.neighbors import NearestNeighbors

ROOT_FOLDER = os.path.dirname(os.getcwd())
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
sys.path.insert(0, ROOT_FOLDER + '/examples/')
sys.path.insert(0, ROOT_FOLDER)

from GAE.graph_case_controller import GraphAutoEncoder
import karate
#%% constant declaration
TRAIN = False
MODEL_FILENAME = ROOT_FOLDER+"/data/gae_kar_stability"
RESULTS_FILE = ROOT_FOLDER+"/data/train_kar_stability"

#%% create graph and train embedding
graph = karate.create_karakte_mirror_network({'weight': 'random'}, 
                                             {'label0': 'random', 'label1': 'random'})

gae = GraphAutoEncoder(graph, learning_rate=0.001, support_size=[5, 5], dims=[2, 8, 8, 8, 8],
                       batch_size=1024, max_total_steps=5000, verbose=True, act=tf.nn.tanh)
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

# 7867

#%% calculate distance between corresponding and all pairs
size = 34  # delta in number between two corresponding nodes

def calc_delta(embed):
    delta_corres = sum([sum(np.abs(np.subtract(embed[i, 1:], embed[i+size, 1:]))) for i in range(size)])
    delta_corres = delta_corres / size

    delta_all = 0
    for i in range(size):
        delta_all = delta_all + sum([sum(np.abs(np.subtract(embed[i, 1:], embed[i + size + j, 1:]))) for j in range(size - i)])
    delta_all = delta_all / ((size -1) * size / 2)
    return delta_corres, delta_all

d_corres, d_all = calc_delta(embed)
print(f'delta correspoding: {d_corres}, delta all: {d_all}')

#%% add random edges and calculate the impact on embeddings

# function to create random edge
def add_random_edge(graph, size, number):
    for i in range(number):
        u = random.randint(size, 2 * size - 1)
        vs = [j for j in range(size, 2 * size) if j not in graph.neighbors(u)]
        v = random.choice(vs)
        w = random.random()
        graph.add_edge(u, v, weight=w)

def get_nearest_neighbour(embed, size):
    """
    Retrieves the nearest neightbour of the id in the range of nodes with id between 0 and size.
    The ID is expected to be greater then the size. 
    Size is the size of the single karate network, excl the mirrored part
    """
    # calculate nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(embed[:size-1, 1:])
    distances, indices = nbrs.kneighbors(embed[size:size * 2 - 1, 1:])
    # get embed of nearest neighbors
    nn_embed = [embed[i[0], 1:] for i in indices]
    # check if nearest neighbour embedding is different from pair embedding.
    count_dif = np.count_nonzero(np.sum(embed[:33, 1:] - nn_embed, axis=1))
    return size - count_dif


sample_size = 25  # number of time the experiment is repeated.
max_noise_level = 25 # maximum number of edges added
edge_noise_corres = []
edge_noise_all = []
edge_noise_nn = []
random.seed(10)
for n in range(1, max_noise_level):
    delta_corres = 0
    delta_all = 0
    delta_nn = 0
    for s in range(sample_size):
        Gtest = graph.copy()
        # add random edges
        add_random_edge(Gtest, size, n)
        embed = gae.calculate_embeddings(graph=Gtest, verbose=False)
        # check how many have a other nearest neighbor then their pair
        cnt_dif = get_nearest_neighbour(embed, size)
        delta_nn = delta_nn + cnt_dif
        # check average delta for pairs and all combinations
        d_corres, d_all = calc_delta(embed)
        delta_corres = delta_corres + d_corres
        delta_all = delta_all + d_all

    delta_nn = delta_nn / sample_size / size  # incorrect percentage NN
    delta_corres = delta_corres / sample_size / 8 # delta per embedding dimension
    delta_all = delta_all / sample_size / 8 # delta per embedding dimention for all combinations
    edge_noise_nn.append(delta_nn)
    edge_noise_corres.append(delta_corres)
    edge_noise_all.append(delta_all)
    print(f'noise level {n} with delta correspoding: {delta_corres}, delta all: {delta_all}, correct NN {delta_nn} ')

plt.plot(edge_noise_all, label='all')
plt.plot(edge_noise_corres, label='pair')
plt.plot(edge_noise_nn, label='% correct nn')
plt.legend()
plt.title("Impact of noise from additional edges")
plt.xlabel("number of additional edges")
plt.ylabel("avg delta in loss")
plt.show()

#%% Measure noise due to changes in Node labels
def change_node_labels(graph, size, n):
    """
    changes n random node labels in graph for nodes with id > size.
    """
    for i in range(n):
        node_id = random.choice(range(size))  # select random node
        node_lbl = graph.nodes[node_id]  # retrieve labels
        lbl = random.choice(list(node_lbl.keys()))  # select random label
        node_lbl[lbl] = random.random()  # set new random value


sample_size = 25  # number of time the experiment is repeated.
max_noise_level = 25 # maximum number of edges added
node_noise_corres = []
node_noise_all = []
node_noise_nn = []
random.seed(10)
for n in range(1, max_noise_level):
    delta_corres = 0
    delta_all = 0
    delta_nn = 0
    for s in range(sample_size):
        Gtest = graph.copy()
        change_node_labels(Gtest, size, n)  # change n random node labels
        embed = gae.calculate_embeddings(graph=Gtest, verbose=False)
        # check how many have a other nearest neighbor then their pair
        cnt_dif = get_nearest_neighbour(embed, size)
        delta_nn = delta_nn + cnt_dif
        # check average delta for pairs and all combinations
        d_corres, d_all = calc_delta(embed)
        delta_corres = delta_corres + d_corres
        delta_all = delta_all + d_all

    delta_nn = delta_nn / sample_size / size  # incorrect percentage NN
    delta_corres = delta_corres / sample_size / 8 # delta per embedding dimension
    delta_all = delta_all / sample_size / 8 # delta per embedding dimention for all combinations
    node_noise_nn.append(delta_nn)
    node_noise_corres.append(delta_corres)
    node_noise_all.append(delta_all)
    print(f'noise level {n} with delta correspoding: {delta_corres}, delta all: {delta_all}, correct NN {delta_nn} ')

plt.plot(node_noise_all, label='all')
plt.plot(node_noise_corres, label='pair')
plt.plot(node_noise_nn, label='% correct nn')
plt.legend()
plt.title("Impact of noise from node labels")
plt.xlabel("number of changes node labels")
plt.ylabel("avg delta in loss")
plt.show()

#%% Measure noise due to changes in edge label values
def change_edge_label_value(graph, size, n):
    """
    changes n random edge labels in graph for nodes with id > size.
    """
    for i in range(n):
        edges_for_selection = [(x,y) for x, y in graph.edges if x < size and y < size]
        edge_nodes = random.choice(edges_for_selection)  # select random edge
        edge_nodes = (edge_nodes[0] + size, edge_nodes[1] + size)
        out_edge = graph.edges[(edge_nodes[0], edge_nodes[1])]  # retrieve out edge
        in_edge = graph.edges[(edge_nodes[1], edge_nodes[0])]  # retrieve in edge
        out_edge['weight'] = random.random()  # set new random value
        in_edge['weight'] =  random.random()  # set new random value


sample_size = 25  # number of time the experiment is repeated.
max_noise_level = 25 # maximum number of edges added
edge_val_noise_corres = []
edge_val_noise_all = []
edge_val_noise_nn = []
random.seed(10)
for n in range(1, max_noise_level):
    delta_corres = 0
    delta_all = 0
    delta_nn = 0
    for s in range(sample_size):
        Gtest = graph.copy()
        change_edge_label_value(Gtest, size, n)  # change n random node labels
        embed = gae.calculate_embeddings(graph=Gtest, verbose=False)
        # check how many have a other nearest neighbor then their pair
        cnt_dif = get_nearest_neighbour(embed, size)
        delta_nn = delta_nn + cnt_dif
        # check average delta for pairs and all combinations
        d_corres, d_all = calc_delta(embed)
        delta_corres = delta_corres + d_corres
        delta_all = delta_all + d_all

    delta_nn = delta_nn / sample_size / size  # incorrect percentage NN
    delta_corres = delta_corres / sample_size / 8 # delta per embedding dimension
    delta_all = delta_all / sample_size / 8 # delta per embedding dimention for all combinations
    edge_val_noise_nn.append(delta_nn)
    edge_val_noise_corres.append(delta_corres)
    edge_val_noise_all.append(delta_all)
    print(f'noise level {n} with delta correspoding: {delta_corres}, delta all: {delta_all}, correct NN {delta_nn} ')

plt.plot(edge_val_noise_all, label='all')
plt.plot(edge_val_noise_corres, label='pair')
plt.plot(edge_val_noise_nn, label='% correct nn')
plt.legend()
plt.title("Impact of noise from edge label value")
plt.xlabel("number of changed edge labels")
plt.ylabel("avg delta in loss")
plt.show()

# %%
