import unittest
import networkx as nx
import os
print (os.getcwd())
from  GAE.dataFeederNx import DataFeederNx
import examples.example_graph_bell as gb
import tensorflow as tf

class TestDataFeederNx(unittest.TestCase):

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
        self.assertEqual(in_weight.shape, (14, 3))
        self.assertListEqual(in_weight[1].tolist(), [1, 1, 1])
        self.assertListEqual(in_weight[7].tolist(), [0.3, 0, 0]) 

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
        self.assertEqual(out_weight.shape, (14, 3))
        self.assertListEqual(out_weight[1].tolist(), [1, 1, 1])
        self.assertListEqual(out_weight[7].tolist(), [0.3, 0, 0]) 
        self.assertListEqual(out_weight[6].tolist(), [0, 0, 0]) 

    def test_feature_dim(self):
        self.assertEqual(self.data_feeder.feature_dim, 2)

if __name__ == '__main__':
    unittest.main()