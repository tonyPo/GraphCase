# import example_graph_bell_version2  as gb
#%%
import os
import sys
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase/'
sys.path.insert(0, '/Users/tonpoppe/workspace/GraphCase/')

from  GAE.graph_case_controller import GraphAutoEncoder
import example_graph_bell as gb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import random
from pyvis import network as net
#%%
graph = gb.create_directed_barbell(10, 10)
random.seed(2)
for u in graph.nodes(data=True):
    u[1]['label1'] = int(u[0])
    u[1]['label2'] = random.uniform(0.0, 1.0)
gae = GraphAutoEncoder(graph, learning_rate=0.01, support_size=[5, 5], dims=[3, 5, 7, 6, 2],
                       batch_size=12, max_total_steps=10, verbose=True, useBN=True)
gae.fit()
embed = gae.calculate_embeddings()
l1_struct, graph2 = gae.get_l1_structure(15, show_graph=False, node_label='feat0')

#%% Decode embedding
emb1 = embed[1,1:]
feat, nbh, nt = gae.decode(emb1, incl_graph='pyvis')

# nx.draw(nt)
nt.show(ROOT_FOLDER + '/temp/scratch.html')
# %%
