import example_graph_bell  as gb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n=5
G = gb.create_directed_barbell(4, 4)

node_label_stats = {}
for i in G.nodes.keys():
    for lbl in  G.nodes[i].keys():
        node_label_stats[lbl] = node_label_stats.get(lbl, 0) + 1

max_value = max(node_label_stats.values())
incl_list = []
excl_list = []
for k,v in node_label_stats.items():
    if v == max_value:
        incl_list.append(k)
    else:
        excl_list.append(k)

print(incl_list)

features = []
for l in incl_list:
    features.append([x for _,x in sorted(nx.get_node_attributes(G, l).items())])

y = np.array(features)
print(y)
        

# G = create_direted_complete(n)
# plt.subplot(111)
# pos = nx.spring_layout(G)
# pos = nx.kamada_kawai_layout(G)
# color = [x for _,x in sorted(nx.get_node_attributes(G,'label2').items())]
# edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
# print(weights)
# options = {
#     'node_color': color,
#     'node_size': 300,
#     'edgelist':edges, 
#     'edge_color':weights,
#     'width': 2,
#     'with_labels': True,
#     'pos': pos,
#     'edge_cmap': plt.cm.Dark2,
#     'cmap': plt.cm.Dark2
# }
# nx.draw(G, **options)
# plt.show()
