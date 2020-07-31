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
        self.assertEqual(features.shape, (2,13))
        

if __name__ == '__main__':
    unittest.main()