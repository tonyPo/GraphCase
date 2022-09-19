#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9-10-2020

@author: tonpoppe
"""
import math
import networkx as nx
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pyvis import network as net

class GraphReconstructor:
    """
    Class for reconstruction the sampled local neighbourhood into a graph based on the inputlayer
    of the encoder or outputlayer of the decoder. The reconstructed graph is a networkx graph.
    """
    def __init__(self, deduplicate=True, delta=0.0001, dummy=0):
        self.node_dim = 0  # dimension of the node labels
        self.edge_dim = 0  # dimension of the edge labels
        self.support_size = [0] # number of samples per layer
        self.layers = 0  # number of layers
        self.deduplicate = deduplicate
        self.delta = delta
        self.dummy = [dummy]

    def reconstruct_graph(self, target, inputlayer, support_size):
        """
        Reconstrucs the samples local neighbourhood into a graph based on the inputlayer
        of the encoder or outputlayer of the decoder. The reconstructed graph is a networkx graph.

        Args:
            target: numpy array with the features of the target node.
            inputLayer: 2-d numpy array containing the sampled feature and edge values.
            support_size: list containing the number of samples per layer.

        Returns:
            graphx graph consisting of the sampled nodes
        """

        self.node_dim = target.shape[-1]
        self.dummy = self.dummy * self.node_dim
        self.edge_dim = tf.shape(inputlayer)[2].numpy() - self.node_dim
        self.support_size = support_size
        self.layers = len(support_size)

        block_size = self.layers - 1 + self.support_size[-1]
        blocks = tf.shape(inputlayer)[1].numpy() / block_size  # number of blocks

        graph = nx.DiGraph()
        parent_feat = dict([('feat'+str(i), t) for i, t in enumerate(target)])
        graph.add_node(1, **parent_feat)
        for block_nr in range(int(blocks)):
            start_row = block_nr * block_size
            block = inputlayer[:, start_row : start_row + block_size, :]
            self.__process_blocks(graph, 1, block, 1, block_nr/ blocks)

        return graph

    def __process_blocks(self, graph, layer, block, parent, block_nr_ratio):
        # determine the direction by calculation the number of switches per sample.
        switch_cnt = np.prod(self.support_size[:layer - 1]) * 2 ** layer
        layer_id = math.floor(block_nr_ratio * switch_cnt)
        is_incoming = (layer_id % 2) == 0
        # print(f"layer = {layer}, block_rat = {block_nr_ratio}, incoming ={is_incoming}")

        if layer < self.layers:           
            child = self.__add_node_edge(graph, parent, block[:, 0:1, :], is_incoming)
            self.__process_blocks(graph, layer+1, block[:, 1:, :], child, block_nr_ratio)

        else:
            for i in range(tf.shape(block)[1].numpy()):
                self.__add_node_edge(graph, parent, block[:, i:i+1, :], is_incoming)

    def __add_node_edge(self, graph, parent, node_edge, is_incoming=True):
        node = node_edge[0, 0, -self.node_dim:]
        edge = node_edge[0, 0, 0:self.node_dim]
        node_id = self.__add_node(graph, node)
        if node_id != 0: #  node is not a dummy node
            edge_feat = dict([('edge_feat'+str(i), t) for i, t in enumerate(edge.numpy())])
            if is_incoming:
                graph.add_edge(node_id, parent, **edge_feat)
            else:
                graph.add_edge(parent, node_id, **edge_feat)
        return node_id

    def __add_node(self, graph, node):
        new_id = graph.number_of_nodes() + 1

        # check if node matches dummy node.
        equal_count = len([i for i, j in zip(node.numpy(), self.dummy) if abs(i - j) < self.delta])
        if equal_count == self.node_dim:
            return 0

        # check if node is already part of the graph.
        if self.deduplicate:
            for u in graph.nodes(data=True):
                u_feat = [v for k, v in sorted(u[1].items(), key=lambda tup: int(tup[0][4:]))]
                count = len([i for i, j in zip(node.numpy(), u_feat) if abs(i - j) < self.delta])
                if count == self.node_dim:
                    return u[0]

        # add node to graph.
        node_feat = dict([('feat'+str(i), t) for i, t in enumerate(node.numpy())])
        graph.add_node(new_id, **node_feat)
        return new_id

    @staticmethod
    def show_graph(graph, node_label=None, ax=None):
        
        if node_label is not None:
            node_labels = nx.get_node_attributes(graph, node_label)
        else:
            node_labels = None
        pos = nx.kamada_kawai_layout(graph)
        edge_labels = nx.get_edge_attributes(graph, name='edge_feat0')
        length = nx.single_source_dijkstra_path_length(graph.to_undirected(), 1, 2, weight=1)
        color = [v / 2 for k, v in sorted(length.items(), key=lambda tup: int(tup[0]))]
        options = {
            'node_color': color,
            'node_size': 300,
            'width': 1,
            'with_labels': True,
            'labels': node_labels,
            'edge_labels': nx.get_edge_attributes(graph, name='edge_feat0'),
            'pos': pos,
            'cmap': plt.cm.rainbow
        }
        if ax is None:
            nx.draw(graph, **options)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
            plt.show()
        else:
            nx.draw(graph, **options, ax=ax)
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax)

    def show_pyvis(self, graph, node_label=None):
        nt = net.Network(notebook=True, directed=True)
        # nt.from_nx(graph)
        nt.set_edge_smooth('straightCross')
        length_dict = nx.single_source_dijkstra_path_length(graph.to_undirected(), 1, 3, weight=1)
        color_dict = {0: 'red', 1: 'lightblue', 2: 'lightgreen'}
        for node in graph.nodes(data=True):
            if node_label is not None:
                nt.add_node(node[0], str(node[1][node_label]), color=color_dict[length_dict[node[0]]],
                            shape='circle')
            else:
                nt.add_node(node[0], node[0], color=color_dict[length_dict[node[0]]],
                            shape='circle')

        for o, i, l in graph.edges(data=True):
            nt.add_edge(o, i, label=str(round(l['edge_feat0'], 2)))

        return nt