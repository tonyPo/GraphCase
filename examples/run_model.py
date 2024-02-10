#%%
import os
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase/'
os.chdir(ROOT_FOLDER)
SHOW_PLOTS = False
#%%
import os
import sys
sys.path.insert(0, os.getcwd())
import networkx as nx
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from GAE.graph_case_controller import GraphAutoEncoder
from GAE.graph_reconstructor import GraphReconstructor
from  GAE.graph_case_tools import Tools
import examples.example_graph_bell_version2 as gb


#%% Create barbell graph with 2 edge labels and one node label.
G = gb.create_directed_barbell(5, 5)
for u,v,d in G.edges(data=True):
    d['edge_lbl1'] = u/v + 0.011

for i in G.nodes():
    G.nodes[i]['target']=1 if (i>4) and (i<10) else 0

if SHOW_PLOTS:
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

import networkx as nx
with open("/Users/tonpoppe/Downloads/graph.pickle", 'rb') as f:
    G = pickle.load(f)


gae = GraphAutoEncoder(
    G, support_size=[4, 4], dims=[3, 16, 16, 16], batch_size=3, hub0_feature_with_neighb_dim=None,
    useBN=True, verbose=True, seed=1, learning_rate=0.002, act=tf.nn.relu, encoder_labels=['attr1', 'attr2']
)

#%% Create Graph auto encoder and train it on the barbel graph
# gae = GraphAutoEncoder(
#     G, support_size=[4, 4], dims=[3, 16, 16, 16], batch_size=3, hub0_feature_with_neighb_dim=16,
#     useBN=True, verbose=True, seed=1, learning_rate=0.002, act=tf.nn.relu, encoder_labels=['label1']
# )


history = gae.fit(epochs=3, layer_wise=False)
if SHOW_PLOTS:
    plt.plot(history[None].history['loss'], label='loss')
    plt.plot(history[None].history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

#%% Calculate the embedings of the nodes
e = gae.calculate_embeddings(G)

if SHOW_PLOTS:
    fig, ax = plt.subplots(1,2, figsize=(20,5))
    ax[0].scatter(e[:,1], e[:,2], s=200., c=color, cmap=plt.cm.Dark2)
    for i, txt in enumerate(e[:,0]):
        ax[0].annotate(txt, (e[i,1], e[i,2]))
    ax[0].set_xlabel("Leprechauns")
    ax[0].set_ylabel("Gold")

    nx.draw(G, **options, ax=ax[1])
    plt.title("Barbel graph: node coler represents the out_degree, label = node id")
    plt.show()

#%%
# # save and restore model
# gae.save_model("saved_model")
# gae_new1 = GraphAutoEncoder.load_model("saved_model")

# try:
#     gae_new1.calculate_embeddings(G)
# except:
#     print("restored model can only be called")

# gae.save_weights("saved_model/saved_weights")
# gae_new2 = GraphAutoEncoder(G, support_size=[3, 3], dims=[2, 6, 6, 4], batch_size=3,
#                         hub0_feature_with_neighb_dim=2, useBN=True, encoder_labels=['label1'],
#                        verbose=False, seed=3, learning_rate=0.002)
# gae_new2.load_weights("saved_model/saved_weights")
# e2 = gae_new2.calculate_embeddings(G)
# print(f"Both models have the same results: {(e==e2).all()}")

#%% decode an embedding back into a subgraph
feat_out, df_out, recon_graph = gae.decode(e[0,1:], incl_graph='graph')
print(f"features (one node feature) of the node has shape: {feat_out.shape}")
print(f"The local neighbourhood has shape: {df_out.shape}")
print(f"2 (in + out neighbourhood) * 6 (level 1 in/out) * 4( level 2 + 1 one parent in/out) = 48")
print(f"1 node feature and 2 edge features")

if SHOW_PLOTS:
    GraphReconstructor.show_graph(recon_graph)

# using pyvis
feat, nbh, nt = gae.decode(e[0,1:], incl_graph='pyvis')
nt.show(ROOT_FOLDER + '/temp/scratch.html')
# results need to be view in the browser!

# %% Show input layer
l1_struct, graph2 = gae.get_l1_structure(7, show_graph=False, node_label='feat0', deduplicate=False)

print(f"l1 structure has shape {l1_struct.shape}")

if SHOW_PLOTS:
    fig, ax = plt.subplots(1,2, figsize=(20,5))
    im = Tools.plot_layer(np.squeeze(l1_struct), 10)
    ax[0].imshow(im)

    GraphReconstructor.show_graph(graph2, ax=ax[1])

    plt.show()

# %% supervised model

supervised_part = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu'), 
    tf.keras.layers.Dense(1)
])

compiler_dict={
    'loss': tf.keras.losses.binary_crossentropy,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=0.005)
}

train_dict={
    'epochs': 6
}
history = gae.fit_supervised('target', supervised_part, compiler_dict, train_dict, verbose=True)

if SHOW_PLOTS:
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
print(history.history['loss'])