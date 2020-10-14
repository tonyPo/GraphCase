# import example_graph_bell_version2  as gb
#%%
import os
import sys
sys.path.insert(0, '/Users/tonpoppe/workspace/GraphCase/')

from  GAE.graph_case_controller import GraphAutoEncoder
import example_graph_bell as gb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
import random
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
l1_struct, graph2 = gae.get_l1_structure(15, show_graph=True, node_label='feat0')

#%%

# print(l1_struct)
# train_res = {}
# for i in range(len(gae.dims)):
#     train_res["l"+str(i+1)] = gae.train_layer(i+1)

# train_res['all'] = gae.train_layer(len(gae.dims), all_layers=True, dropout=None)
# embed = gae.calculate_embeddings()
# filename = '/Users/tonpoppe/workspace/GraphCase/data/model1'
# gae.save_model(filename)

# gae2 = GraphAutoEncoder(graph, learning_rate=0.01, support_size=[5, 5], dims=[3, 5, 7, 6, 2],
#                        batch_size=12, max_total_steps=100, verbose=True)
# gae2.load_model(filename, graph)
# embed2 = gae2.calculate_embeddings()

# embed3 = np.subtract(embed, embed2)
# print(embed3)


# gae.load_model('/Users/tonpoppe/workspace/GraphCase/data/model1')


#%%

# ad node ids to the graph as label
# labels3 = [(i, i) for i in range(13)]
# labels3 = dict(labels3)
# nx.set_node_attributes(graph, labels3, 'label3')

# gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
#                                max_total_steps=1, verbose=True, seed=2)

# gae = GraphAutoEncoder(graph, support_size=[3, 4, 5], dims=[2, 3, 3, 3, 3, 2], batch_size=3,
#                                max_total_steps=1, verbose=True, seed=2)
# h = gae.train_layer(1)
# print(h)
# print(f"enc weights \n {gae.model.layer_enc[1].weights}")
# print(f"dec weights \n {gae.model.layer_dec[1].weights}")



# print(f"in_weight_list: {in_weight_list}")
# order by weight
# cust by length
# add dummy
# convert to list

# node_label_stats = {}
# for i in G.nodes.keys():
#     for lbl in  G.nodes[i].keys():
#         node_label_stats[lbl] = node_label_stats.get(lbl, 0) + 1

# max_value = max(node_label_stats.values())
# incl_list = []
# excl_list = []
# for k,v in node_label_stats.items():
#     if v == max_value:
#         incl_list.append(k)
#     else:
#         excl_list.append(k)

# print(incl_list)

# features = []
# for l in incl_list:
#     features.append([x for _,x in sorted(nx.get_node_attributes(G, l).items())])

# y = np.array(features)
# print(y)
        

# G = create_direted_complete(n)
# plt.subplots(111)
# pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(graph)
# # color = [x for _,x in sorted(nx.get_node_attributes(G,'label1').items())]
# color = [graph.out_degree(x) for x in range(graph.number_of_nodes())]
# edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
# # print(weights)
options = {
    # 'node_color': color,
    'node_size': 100,
    # 'edgelist':edges, 
    # 'edge_color':weights,
    'width': 1,
    'with_labels': True,
    'pos': pos,
    'edge_cmap': plt.cm.Dark2,
    'cmap': plt.cm.Dark2
}
nx.draw(graph, **options)
plt.show()
#%%
# 555 dropout 0.3

# 490
# 686
# 632
# train_res = pickle.load(open("/Users/tonpoppe/train_res3", "rb"))
# plt.subplot(111)
# plt.plot(train_res['all']['i'], train_res['all']['val_l'], label='all')
# plt.plot(train_res['l1']['i'], train_res['l1']['val_l'], label='l1')
# plt.plot(train_res['l2']['i'], train_res['l2']['val_l'], label='l2')
# plt.plot(train_res['l3']['i'], train_res['l3']['val_l'], label='l3')
# plt.plot(train_res['l4']['i'], train_res['l4']['val_l'], label='l4')
# plt.plot(train_res['l5']['i'], train_res['l5']['val_l'], label='l5')
# plt.legend()
# plt.yscale('log')
# plt.xlabel("iteration")
# plt.ylabel("validaiton loss")

# plt.show()

# %%
