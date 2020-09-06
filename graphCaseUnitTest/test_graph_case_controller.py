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
        labels3 = dict(labels3)
        nx.set_node_attributes(graph, labels3, 'label3')
        gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
                               max_total_steps=1, verbose=False, seed=2)
        res = gae.train_layer(1)
        self.assertAlmostEqual(res['l'][0], 2158.0686, 4,
                               "loss of the initial setup does not match with expectations")

        res = gae.train_layer(2)
        self.assertAlmostEqual(res['l'][0], 2613.2725, 4,
                               "loss of the initial setup does not match with expectations")

        res = gae.train_layer(3)
        self.assertAlmostEqual(res['l'][0], 2693.6736, 4,
                               "loss of the initial setup does not match with expectations")

        res = gae.train_layer(4)
        self.assertAlmostEqual(res['l'][0], 2842.3582, 4,
                               "loss of the initial setup does not match with expectations")

        res = gae.train_layer(4, all_layers=True)
        self.assertAlmostEqual(res['l'][0], 2842.1409, 4,
                               "loss of the initial setup does not match with expectations")


    def test_train_layer2(self):
        """
        Test if the loss is reduced during training
        """
        graph = gb.create_directed_barbell(4, 4)
        gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
                               max_total_steps=10, verbose=False, seed=2)
        res = gae.train_layer(1, learning_rate=0.0001)
        self.assertTrue(res['val_l'][0] > res['val_l'][-1],
                        "loss has not decreased while training layer 1")

        res = gae.train_layer(2, learning_rate=0.0001)
        self.assertTrue(res['val_l'][0] > res['val_l'][-1],
                        "loss has not decreased while training layer 2")

        res = gae.train_layer(3, learning_rate=0.0001)
        self.assertTrue(res['val_l'][0] > res['val_l'][-1],
                        "loss has not decreased while training layer 3")

        res = gae.train_layer(4, learning_rate=0.0001)
        self.assertTrue(res['val_l'][0] > res['val_l'][-1],
                        "loss has not decreased while training layer 4")

    def test_train_layer3(self):
        """
        Test with 3 hubs sampling using different support sizes per layer.
        """
        graph = gb.create_directed_barbell(4, 4)
        gae = GraphAutoEncoder(graph, support_size=[3, 4, 5], dims=[2, 3, 3, 3, 3, 2], batch_size=3,
                               max_total_steps=1, verbose=False, seed=2)

        exp = [153.83647, 309.56152, 311.00153, 459.34726, 484.33817, 504.59387]
        for i in range(6):
            res = gae.train_layer(i+1)
            self.assertAlmostEqual(res['l'][0], exp[i], 4,
                                   f"loss of layer {i+1} does not match with expectations")

        res = gae.train_layer(6, all_layers=True)
        self.assertAlmostEqual(res['l'][0], 504.55478, 4,
                               "loss of the layer 6 all traning does not match with expectations")

    def test_train_layer4(self):
        """
        Test using multiple edge label icw a custom weight label. The test checks if the
        weights are calculated correct.
        """
        graph = gb.create_directed_barbell(4, 4)
        for in_node, out_node, lbl in graph.edges(data=True):
            lbl['edge_lbl1'] = in_node/(out_node + 0.011) + 0.22

        gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2], batch_size=3,
                               max_total_steps=10, verbose=False, seed=2, weight_label='edge_lbl1')
        res = gae.train_layer(1, learning_rate=0.0001)
        self.assertAlmostEqual(res['l'][0], 49.392754, 4,
                               "loss of the layer 1 does not match with expectations using a \
                               custom edge label")

    def test_train_layer5(self):
        """
        Test using final combination layer. Test if training works correctly and if the calculation
        of the embeddings works correctly.
        """
        graph = gb.create_directed_barbell(4, 4)
        for in_node, out_node, lbl in graph.edges(data=True):
            lbl['edge_lbl1'] = in_node/(out_node + 0.011) + 0.22

        gae = GraphAutoEncoder(graph, support_size=[3, 3], dims=[2, 3, 3, 2, 2], batch_size=3,
                               max_total_steps=10, verbose=False, seed=2, weight_label='edge_lbl1')


        for i in range(len(gae.dims)):
            res = gae.train_layer(i+1, act=tf.nn.relu)

        self.assertAlmostEqual(res['l'][0], 134.9637, 4,
                               "loss of the last layer does not match with expectations using a \
                               final combination layer")

        res = gae.train_layer(len(gae.dims), all_layers=True, act=tf.nn.relu)
        embed = gae.calculate_embeddings()
        self.assertAlmostEqual(embed[0][2], 38.28431701660156, 4,
                               "embedding of the first batch node differs from expected value")
   