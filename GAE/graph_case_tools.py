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
from PIL import Image
import matplotlib.pyplot as plt

class Tools:
    """ class for tools on GraphCASE
    """
    @staticmethod
    def plot_node(graph, node_id):
        """ sttic method to plot the 2 hub neighbourhood of a node in the graph.
        Args:
            graph:  graph containing the node to plot.
            node_id: The id of the node to plot.
        
        """
        und_graph = graph.to_undirected().copy()
        local_graph = [node_id] + list(und_graph.neighbors(node_id))
        for neightbor in und_graph.neighbors(node_id):
            local_graph = local_graph + [n for n in und_graph.neighbors(neightbor)]
        local_graph = list(set(local_graph))  # make list unique
        subgraph = graph.subgraph(local_graph)

        # plot subgraph
        nt = net.Network(notebook=True, directed=True)
        nt.from_nx(subgraph)
        # nt.set_edge_smooth('straightCross')
        length_dict = nx.single_source_dijkstra_path_length(und_graph, node_id, 2, weight=lambda u, v, d: 1)
        color_dict = {0: 'red', 1: 'lightblue', 2: 'lightgreen'}
        for node in nt.nodes:
            node["color"] = color_dict[length_dict[node['id']]]
            node['shape'] = 'circle'
        for edge in nt.edges:
            edge['label'] = round(edge['weight'], 2)
        nt.toggle_physics(False)
        return nt

    @staticmethod
    def plot_layer(features, size):
        """
        function to visualise a layer where each value is represented with a size x size format.
        
        Args:
            features:   2-d numpy array containing one row per node and for every node the node 
                        and corresponding edge properties.
            size:       pixel size, the number of horizontal and vertical pixels used to
                        visualize one value.

            A png of equal size of the feature numpy x size.
        """

        #expend the pixels width and height to size-value
        pixels = np.repeat(features, size, axis=1)
        pixels = np.repeat(pixels, size, axis=0)

        # rescale between 0 and 255
        pixels = pixels * 255
        pixels = pixels.astype(np.uint8)

        im = Image.fromarray(pixels)

        return im 
    
    @staticmethod    
    def reconstruction_l1(l1_struct, df_out, cutoff=0.1):
        delta = np.abs(l1_struct - df_out)
        filtered_delta = np.where(delta > cutoff, delta, 0)
        fig, ax = plt.subplots(1,4, figsize=(4,10), gridspec_kw={'width_ratios': [3,3,3,3.75]})
        im_orig = Tools.plot_layer(np.squeeze(l1_struct), 10)
        im_recon = Tools.plot_layer(np.squeeze(np.abs(df_out)), 10)
        im_delta = Tools.plot_layer(np.squeeze(delta), 10)
        im_filtered_delta = Tools.plot_layer(np.squeeze(filtered_delta), 10)

        ax[0].imshow(im_orig, vmin=0, vmax=255)
        ax[0].set_title("orig")
        ax[1].imshow(im_recon, vmin=0, vmax=255)
        ax[1].set_title("recon")
        ax[2].imshow(im_delta, vmin=0, vmax=255)
        ax[2].set_title("delta")
        im_delta = ax[3].imshow(im_filtered_delta, vmin=0, vmax=255)
        ax[3].set_title("filtered_delta")

        ax[1].set_axis_off()
        ax[2].set_axis_off()
        ax[3].set_axis_off()
        ax[0].set_yticks(np.arange(0, 240, step= 30),['inc_1-inc','inc_1-outg','inc_2-inc','inc_2-outg','outg_1-inc','outg_1-outg','outg_2-inc','outg_2-outg'])
        ax[0].set_xticks(ticks=[])

        fig.colorbar(im_delta, ax=ax[3])
        plt.show()