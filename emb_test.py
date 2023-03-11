#%%
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from GAE.graph_case_controller import GraphAutoEncoder
from GAE.graph_reconstructor import GraphReconstructor
from GAE.input_layer_constructor import InputLayerConstructor
from  GAE.graph_case_tools import Tools
import examples.example_graph_bell_version2 as gb
import math

# %%
G =  nx.read_gpickle("/Users/tonpoppe/Downloads/graph.pickle")

plt.subplot(111)
# # pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(G)
color = [G.out_degree(x) for x in range(G.number_of_nodes())]
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
options = {
'node_color': color,
'node_size': 300,
'edgelist':edges,
'edge_color':weights,
'width': 1,
'with_labels': True,
'pos': pos,
'edge_cmap': plt.cm.Dark2,
'cmap': plt.cm.Dark2
}
nx.draw(G, **options)
plt.title("Barbel graph: node coler represents the out_degree, label = node id")
plt.show()
# %%

gae = GraphAutoEncoder(
    G, support_size=[4, 4], dims=[3, 16, 16, 16], batch_size=3, hub0_feature_with_neighb_dim=16,
    useBN=True, verbose=True, seed=1, learning_rate=0.002, act=tf.nn.relu, encoder_labels=['attr1', 'attr2']
)
# %%
in_sample = gae.sampler.data_feeder.in_sample
out_sample = gae.sampler.data_feeder.out_sample
# spd = gae.sampler.position_manager.single_pos_dict

# %%
Z = gae.sampler.get_input_layer(tf.constant([0,1,3]), 1)
Z[0].shape
Z[1].shape

# %%
history = gae.fit(epochs=3, layer_wise=False)
# %%
