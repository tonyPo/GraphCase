#%%
import os
os.chdir('..')

# %%
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from GAE.graph_case_controller import GraphAutoEncoder
from GAE.graph_reconstructor import GraphReconstructor
from GAE.input_layer_constructor import InputLayerConstructor
from GAE.position_manager import WaveLetPositionManager
from GAE.position_manager import PositionManager
from  GAE.graph_case_tools import Tools
import examples.example_graph_bell_version2 as gb
import math
# %%
import networkx as nx
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
plt.title("Barbel graph: label = node id")
plt.show()
# %%
# gae2 = GraphAutoEncoder(
#     G, support_size=[4, 4], dims=[3, 16, 16, 16], batch_size=3, hub0_feature_with_neighb_dim=None,
#     useBN=True, verbose=True, seed=1, learning_rate=0.002, act=tf.nn.relu, encoder_labels=['attr1', 'attr2'],
#     pos_enc_cls=PositionManager
# )


# # %%
# G = nx.read_gpickle("/Users/tonpoppe/workspace/graphcase_exp_pos_encoding_rep/graphcase_exp_pos_encoding/graphs/enron_sub_graph4.pickle")
# print(f"number of nodes {G.number_of_nodes()}")
# print(f"number of edges {G.number_of_edges()}")
# #%%
# gae2 = GraphAutoEncoder(
#     G, support_size=[4, 4], dims=[3, 16, 16, 16], batch_size=3, hub0_feature_with_neighb_dim=None,
#     useBN=True, verbose=True, seed=1, learning_rate=0.002, act=tf.nn.relu, encoder_labels=['attr_received_size', 'attr_cnt_to'],
#     pos_enc_cls=WaveLetPositionManager
# )
# %%

from GAE.graph_case_controller import GraphAutoEncoder
from GAE.graph_reconstructor import GraphReconstructor
from GAE.input_layer_constructor import InputLayerConstructor
from  GAE.graph_case_tools import Tools
import examples.example_graph_bell_version2 as gb
import math
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql import functions as F
#create Spark instance
if 'spark' not in (locals() or globals()):
    conf = SparkConf().setAppName('appName').setMaster('local')
    sc = SparkContext(conf=conf)
    global spark 
    spark = SparkSession(sc)

# %%
import pandas as pd
def create_node_df(G):
    nodes = G.nodes(data=True)
    pdf = pd.DataFrame([[k] + list(v.values()) for k,v in nodes], columns= ['id', 'attr1', 'attr2', 'label'])
    return spark.createDataFrame(pdf)

def create_edges_df(G, lbls):
    edges = G.edges(data=True)
    pdf = pd.DataFrame([[s, d]+list(a.values()) for (s,d,a) in edges],
                        columns=lbls
    )
    return spark.createDataFrame(pdf)  

nodes = create_node_df(G)
edges = create_edges_df(G, ['src', 'dst', 'weight'])
graph = (nodes, edges)
# %%
from GAE.position_manager import PositionManager
from GAE.data_feeder_graphframes import DataFeederGraphFrames
gae = GraphAutoEncoder(
    graph, support_size=[5, 5], dims=[6, 8, 8, 8], batch_size=3, hub0_feature_with_neighb_dim=8, encoder_labels=['attr1', 'attr2'],
    useBN=True, verbose=True, seed=1, learning_rate=0.001, act=tf.nn.sigmoid, pos_enc_cls=PositionManager,
    data_feeder_cls=DataFeederGraphFrames
)
# %%
history = gae.fit(epochs=20, layer_wise=False)

plt.plot(history[None].history['loss'], label='loss')
plt.plot(history[None].history['val_loss'], label='val_loss')
plt.legend()
plt.show()
# %%
NODE_ID = 24

l1_struct, graph2 = gae.get_l1_structure(NODE_ID, show_graph=False, node_label='node_id', deduplicate=True)
print(f"l1 structure has shape {l1_struct.shape}")
nt = GraphReconstructor().show_graph(graph2)
# %%
embed = gae.calculate_embeddings()
# %%
embedding = embed[embed[:,0]==NODE_ID].flatten()[1:]
root_feature, reconstruced_l1, recon_graph = gae.decode(embedding, incl_graph='graph', delta=0.3, fraction_sim=1)
GraphReconstructor().show_graph(recon_graph)
# %%
Tools.reconstruction_l1(l1_struct, reconstruced_l1, cutoff=0.2)


# %%