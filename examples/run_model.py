import os
import sys
sys.path.insert(0, os.getcwd())
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from GAE.graph_case_controller import GraphAutoEncoder
import examples.example_graph_bell_version2 as gb


G = gb.create_directed_barbell(4, 4)
for u,v,d in G.edges(data=True):
        d['edge_lbl1'] = u/v + 0.011
# r = G.out_edges(data=True)

# in_edges_dict = {}
# in_weight_dict = {}
# for out_node, in_node, weight in G.in_edges(data=True):
#         in_edges_dict[in_node] = in_edges_dict.get(in_node, list()) + \
#                             [(out_node, list(weight.values()))]
#         in_weight_dict[in_node] = in_weight_dict.get(in_node, list()) + \
#                             [(out_node, weight['weight'])]


# print(in_edges_dict)
# print(in_weight_dict)
gae = GraphAutoEncoder(G, support_size=[3, 4], dims=[2, 6, 6, 2, 1], batch_size=5,
                       max_total_steps=10, verbose=True, seed=2)

for i in range(len(gae.dims)):
    h = gae.train_layer(i+1, act=tf.nn.relu)

h = gae.train_layer(len(gae.dims), all_layers=True, act=tf.nn.relu)
# # print(h1['val_l'])

e = gae.calculate_embeddings()
print(f"e: \n {e}")

# fig, ax = plt.subplots()
# ax.scatter(e[:,1], e[:,2])
# for i, txt in enumerate(e[:,0]):
#     ax.annotate(txt, (e[i,1], e[i,2]))
# plt.xlabel("Leprechauns")
# plt.ylabel("Gold")
# plt.legend(loc='upper left')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(h['i'], h['val_l'])
# plt.xlabel("iteration")
# plt.ylabel("validaiton loss")
# plt.show()

# plt.subplot(111)
# # # pos = nx.spring_layout(G)
# pos = nx.kamada_kawai_layout(G)
# # # color = [x for _,x in sorted(nx.get_node_attributes(G,'label1').items())]
# color = [G.out_degree(x) for x in range(G.number_of_nodes())]
# edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
# # # print(weights)
# options = {
#     'node_color': color,
#     'node_size': 300,
#     'edgelist':edges,
#     'edge_color':weights,
#     'width': 1,
#     'with_labels': True,
#     'pos': pos,
#     'edge_cmap': plt.cm.Dark2,
#     'cmap': plt.cm.Dark2
# }
# nx.draw(G, **options)
# plt.show()
