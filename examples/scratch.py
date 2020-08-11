import example_graph_bell_version2  as gb
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

n=5
G = gb.create_directed_barbell(10, 10)

# get tuple (node, weight) per in _list per node
edges = G.out_edges(data=True)
print(edges)
in_edges_dict = {}
for o, i, w in edges:
    in_edges_dict[o] = in_edges_dict.get(o, list())+ [(i,list(w.values())[0])]

print(f"in_edges_dict {in_edges_dict}")
s=2
dummy_id = 20
dummy_weight = 0
# for k,v in in_edges_dict.items():
nodes = [i for i in range(13)]
for k in nodes:
    v = sorted(in_edges_dict.get(k,[(dummy_id, dummy_weight)]), key = lambda x: x[1], reverse=True )
    if len(v) <= s:
        dummy_cnt =  s - len(v)
        v = v + [(dummy_id, dummy_weight)] * dummy_cnt
    else:
        v = v[0:s] 
    in_edges_dict[k] = v

in_edges_list = []
in_weight_list = []
for  _, v in sorted(in_edges_dict.items()):
    in_edges_list.append([t[0] for t in v])
    in_weight_list.append([t[1] for t in v])


print(f"in_edges_list: {in_edges_list}")
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
# pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(G)
# color = [x for _,x in sorted(nx.get_node_attributes(G,'label1').items())]
color = [G.out_degree(x) for x in range(G.number_of_nodes())]
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
# print(weights)
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
plt.show()
