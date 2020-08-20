import unittest
import networkx as nx
import os
import sys
from  GAE.graph_case_controller import GraphAutoEncoder
import examples.example_graph_bell as gb
import tensorflow as tf

class TestGraphCaseController(unittest.TestCase):
    """
    Unit test for the class graphCaseController
    """

    def test_consistency_checks(self):
        """
        Test the checks during initializations.
        """
        graph = gb.create_directed_barbell(10, 10)
        with self.assertRaises(AssertionError):
            gae = GraphAutoEncoder(graph, support_size=[5, 5], dims=[2, 6, 6], batch_size=1024,
                                   max_total_steps=10, verbose=True, seed=2)

    def test_train_layer(self):
        """
        Test if the loss of the initial setup is correct.
        """
        graph = gb.create_directed_barbell(4, 4)
        # ad node ids to the graph as label
        labels3 = [(i, i) for i in range(13)]
        labels3 = dict(labels3  )
        nx.set_node_attributes(graph, labels3, 'label3')
        gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
                               max_total_steps=1, verbose=False, seed=2)
        h = gae.train_layer(1)
        self.assertAlmostEqual(h['l'][0], 2158.0686, 4,
                               "loss of the initial setup does not match with expectations")

        h = gae.train_layer(2)
        self.assertAlmostEqual(h['l'][0], 2613.2725, 4,
                               "loss of the initial setup does not match with expectations")

        h = gae.train_layer(3)
        self.assertAlmostEqual(h['l'][0], 2693.6736, 4,
                               "loss of the initial setup does not match with expectations")

        h = gae.train_layer(4)
        self.assertAlmostEqual(h['l'][0], 2842.3582, 4,
                               "loss of the initial setup does not match with expectations")

        h = gae.train_layer(4, all_layers=True)
        self.assertAlmostEqual(h['l'][0], 2842.1409, 4,
                               "loss of the initial setup does not match with expectations")


    def test_train_layer2(self):
        """
        Test if the loss is reduced during training
        """
        graph = gb.create_directed_barbell(4, 4)
        gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
                               max_total_steps=10, verbose=False, seed=2)
        h = gae.train_layer(1, learning_rate=0.0001)
        self.assertTrue(h['val_l'][0] > h['val_l'][-1],
                        "loss has not decreased while training layer 1")

        h = gae.train_layer(2, learning_rate=0.0001)
        self.assertTrue(h['val_l'][0] > h['val_l'][-1],
                        "loss has not decreased while training layer 2")

        h = gae.train_layer(3, learning_rate=0.0001)
        self.assertTrue(h['val_l'][0] > h['val_l'][-1],
                        "loss has not decreased while training layer 3")
        
        h = gae.train_layer(4, learning_rate=0.0001)
        self.assertTrue(h['val_l'][0] > h['val_l'][-1],
                        "loss has not decreased while training layer 4")

