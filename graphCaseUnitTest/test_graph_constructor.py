import unittest
import networkx as nx
import os
import sys
import numpy as np
from  GAE.graph_case_controller import GraphAutoEncoder
import examples.example_graph_bell as gb
import tensorflow as tf
import random

class TestGraphCaseController(unittest.TestCase):
    """
    Unit test for the class graphReconstructor
    """

    def test_reconstruct_graph(self):
        """
        Test the reconstruction of an inputlayer.
        """
        graph = gb.create_directed_barbell(10, 10)
        random.seed(2)
        for u in graph.nodes(data=True):
            u[1]['label1'] = int(u[0])
            u[1]['label2'] = random.uniform(0.0, 1.0)
        gae = GraphAutoEncoder(graph, learning_rate=0.01, support_size=[5, 5], dims=[3, 5, 7, 6, 2],
                               batch_size=12, max_total_steps=100, verbose=True)

        l1_struct, graph2 = gae.get_l1_structure(15, show_graph=False)
        # check if the nodes of the reconstructed graph is equal to 5
        self.assertEqual(graph2.number_of_nodes(), 5,
                         "Number of nodes in reconstructed graph does not match with expectations")

        # check if the returned nodes are correct by summing the node values.
        sum_values = np.sum(l1_struct, 1)
        self.assertAlmostEqual(sum_values[0, 1], 120, 4,
                               "sum of nodes ids in reconstructed graph does not match with expectations")
        self.assertAlmostEqual(sum_values[0, 0], 2.399999, 4,
                               "sum of edges in reconstructed graph does not match with expectations")
