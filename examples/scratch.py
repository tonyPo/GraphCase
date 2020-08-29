# import example_graph_bell_version2  as gb
import os
import sys
sys.path.insert(0, os.getcwd())

from  GAE.graph_case_controller import GraphAutoEncoder
import example_graph_bell as gb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

graph = gb.create_directed_barbell(4, 4)

# ad node ids to the graph as label
labels3 = [(i, i) for i in range(13)]
labels3 = dict(labels3)
nx.set_node_attributes(graph, labels3, 'label3')

gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
                               max_total_steps=1, verbose=True, seed=2)
h = gae.train_layer(1)
print(h)
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
plt.subplot(111)
# # pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(graph)
# # color = [x for _,x in sorted(nx.get_node_attributes(G,'label1').items())]
color = [graph.out_degree(x) for x in range(graph.number_of_nodes())]
edges,weights = zip(*nx.get_edge_attributes(graph,'weight').items())
# # print(weights)
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
nx.draw(graph, **options)
plt.show()
