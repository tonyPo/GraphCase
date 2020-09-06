import unittest
import networkx as nx
import os
import sys
from  GAE.data_feeder_nx import DataFeederNx
import examples.example_graph_bell as gb
import tensorflow as tf

class TestDataFeederNx(unittest.TestCase):
    """
    Unit test for the class DataFeederNx
    """

    # def setUp
    def setUp(self):
        G = gb.create_directed_barbell(4, 4)
        self.data_feeder = DataFeederNx(G)

    def test_init_train_batch(self):
        self.data_feeder.init_train_batch()
        self.assertIsInstance(self.data_feeder.iter["train"], tf.data.Dataset)

    def test_get_valid_node_labels(self):
        features = self.data_feeder.features
        self.assertEqual(features.shape, (14,2))


    def test_get_in_sample(self):
        in_sample = self.data_feeder.in_sample
        self.assertEqual(in_sample.shape, (14, 3))
        self.assertListEqual(in_sample[1].tolist(), [0, 2, 3])
        self.assertListEqual(in_sample[7].tolist(), [8, 13, 13])
        self.assertListEqual(in_sample[13].tolist(), [13, 13, 13])


    def test_get_in_sample_weight(self):
        in_weight = self.data_feeder.in_sample_weight
        self.assertEqual(in_weight.shape, (14, 3, 1))
        self.assertListEqual(in_weight[1].tolist(), [[1], [1], [1]])
        for act, exp in zip(in_weight[7].tolist(), [[0.3], [0], [0]]):
            self.assertAlmostEqual(act[0], exp[0], 4)

    def test_get_out_sample(self):
        out_sample = self.data_feeder.out_sample
        self.assertEqual(out_sample.shape, (14, 3))
        print(out_sample[0])
        self.assertListEqual(out_sample[1].tolist(), [0, 2, 3])
        self.assertListEqual(out_sample[7].tolist(), [6, 13, 13])
        self.assertListEqual(out_sample[6].tolist(), [13, 13, 13])
        self.assertListEqual(out_sample[13].tolist(), [13, 13, 13])

    def test_get_out_sample_weight(self):
        out_weight = self.data_feeder.out_sample_weight
        self.assertEqual(out_weight.shape, (14, 3, 1))
        self.assertListEqual(out_weight[1].tolist(), [[1], [1], [1]])
        for act, exp in zip(out_weight[7].tolist(), [[0.3], [0], [0]]):
            self.assertAlmostEqual(act[0], exp[0], 4)
        self.assertListEqual(out_weight[6].tolist(), [[0], [0], [0]])

    def test_feature_dim(self):
        self.assertEqual(self.data_feeder.feature_dim, 2)

    def test_weight_label(self):
        """
        Test if setting the customer weight label works corrects. 
        Checks if the right label is select and put in the first column of the sample weight tensors
        and checks if the orders is correctly exact on the custom weight label values.
        """
        graph = gb.create_directed_barbell(4, 4)
        edge_weight = [1, 1, 1]
        edge_lbl1_in = [0.22, 2.198239366963403, 3.1873590504451044]
        edge_lbl1_out = [91.12909091, 0.71726504, 0.55211558]
        for in_node, out_node, lbl in graph.edges(data=True):
            lbl['edge_lbl1'] = in_node/(out_node + 0.011) + 0.22
        data_feeders = [DataFeederNx(graph), DataFeederNx(graph, weight_label='edge_lbl1')]
        for nr, data_feeder in enumerate(data_feeders):
            in_weight = data_feeder.in_sample_weight
            out_weight = data_feeder.out_sample_weight
            if nr == 1:
                edge_lbl1_in.sort(reverse=True)

            for i in range(3):
                self.assertAlmostEqual(in_weight[1][i][nr], edge_weight[i], 4)
                self.assertAlmostEqual(in_weight[1][i][abs(nr-1)], edge_lbl1_in[i], 4)

                self.assertAlmostEqual(out_weight[1][i][nr], edge_weight[i], 4)
                self.assertAlmostEqual(out_weight[1][i][abs(nr-1)], edge_lbl1_out[i], 4)



if __name__ == '__main__':
    unittest.main()