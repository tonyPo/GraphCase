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
    G.add_edges_from(edges)
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

    return G


G = create_directed_barbell(4, 4)
plt.subplot(111)
pos = nx.spring_layout(G)
pos = nx.kamada_kawai_layout(G)
options = {
    'node_color': 'lightgreen',
    'node_size': 300,
    'width': 1,
    'with_labels': True,
    'pos': pos
}
nx.draw(G, **options)
plt.show()
