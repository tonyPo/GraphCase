#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 07:26:20 2019

@author: tonpoppe
"""
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import networkx as nx
from GAE.model import GraphAutoEncoderModel
from GAE.data_feeder_nx import DataFeederNx
from GAE.graph_reconstructor import GraphReconstructor
from pyvis import network as net

class Tools:
    @staticmethod
    def plot_node(graph, node_id):
        local_graph = []
        for neightbor in graph.neighbors(node_id):
            local_graph = local_graph + [n for n in graph.neighbors(neightbor)]
        local_graph = list(set(local_graph))  # make list unique
        subgraph = graph.subgraph(local_graph)

        # plot subgraph
        nt = net.Network(notebook=True, directed=True)
        nt.from_nx(subgraph)
        # nt.set_edge_smooth('straightCross')
        length_dict = nx.single_source_dijkstra_path_length(subgraph, node_id, 2, weight=1)
        color_dict = {0: 'red', 1: 'lightblue', 2: 'lightgreen'}
        for node in nt.nodes:
            node["color"] = color_dict[length_dict[node['id']]]
            node['shape'] = 'circle'
        for edge in nt.edges:
            edge['label'] = round(edge['weight'], 2)
        nt.toggle_physics(False)
        return nt
