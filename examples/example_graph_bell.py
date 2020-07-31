#%%
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np


def create_direted_complete(n):
    """
    Return a directed complete graph n nodes having a egde in both
    direction between all nodes.

    :param n: number of nodes
    """
    G = nx.DiGraph()
    edges = itertools.permutations(range(n), 2)
    edges  = [(l, r, 1) for l,r in edges]
    G.add_weighted_edges_from(edges)
    # labels1 = dict([(i, 0.5) if i % 2 == 0 else(i, 0.57) for i in range(n)])
    # labels2 = dict([(i, 1) if i % 2 == 0 else(i, 0) for i in range(n)])
    # nx.set_node_attributes(G, labels1, 'label1')
    # nx.set_node_attributes(G, labels2, 'label2')
    return G


def create_directed_barbell(m1, m2):
    """
    Returns a directed barbell like graph. The bell are directed complete
    graphs having egdes in both directions between every pair of nodes.
    The nodes between the two bells are directed to the center node. Note
    that the path between the two bells needs to be uneven to have a
    center node.

    :paran m1: number of nodes in the bells
    :param m2: number of nodes in the path. Note if this is an even number
                 then an addtional center node is added
    """
    G = create_direted_complete(m1)

    if np.mod(m2, 2) == 0:
        m2 = m2 + 1
    d = int((m2-1)/2+1)

    G.add_edges_from([(v-1, v) for v in range(m1, m1+d)])
    G.add_edges_from([(v, v-1) for v in range(m1+d, m1+m2+1)])

    G_right = create_direted_complete(m1)
    G.add_edges_from((u+m1+m2, v+m1+m2) for (u, v) in list(G_right.edges))

    #add node labels
    labels1_1 = [(i, 0.5) if i % 2 == 0 else(i, 0.75) for i in range(m1)]
    labels1_2 = [(i, 0.5) if i % 2 == 0 else(i, 0.75) for i in range(m1+m2, m1 * 2 +m2)]
    labels1_3 = [(i, 0.25) for i in range(m1, m1 + m2)]
    labels1 = dict(labels1_1 + labels1_2 + labels1_3)
    nx.set_node_attributes(G, labels1, 'label1')

    labels2_1 = [(i, 1) if i % 2 == 0 else(i, 0) for i in range(m1)]
    labels2_2 = [(i, 1) if i % 2 == 0 else(i, 0) for i in range(m1+m2, m1 * 2 +m2)]
    labels2_3 = [(i, 0.5) for i in range(m1, m1 + m2)]
    labels2 = dict(labels2_1 + labels2_2 + labels2_3)

    nx.set_node_attributes(G, labels1, 'label1')
    nx.set_node_attributes(G, labels2, 'label2')

    # add weight
    for u,v,d in G.edges(data=True):
        if (v > m1-1) & (v < m1 + m2):
            d['weight'] = 0.3
        else:
            d['weight'] = 1

    return G


#%%

# n=5
# G = create_directed_barbell(4, 4)
# # G = create_direted_complete(n)
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


# %%
