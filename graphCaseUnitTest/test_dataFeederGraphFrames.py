import unittest
import sys
import random
import pandas as pd
import numpy as np
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
sys.path.insert(0, ROOT_FOLDER)
from GAE.data_feeder_graphframes import spark
from  GAE.data_feeder_graphframes import DataFeederGraphFrames
from  GAE.data_feeder_nx import DataFeederNx
import examples.example_graph_bell as gb
import tensorflow as tf

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

class TestDataFeederGraphFrames(unittest.TestCase):
    """
    Unit test for the class DataFeederNx
    """

    # def setUp
    def setUp(self):
        G = gb.create_directed_barbell(4, 4)
        random.seed(1)
        edge_labels = ['weight', 'edge_lbl1', 'edge_lbl2']
        for _, _, edge in G.edges(data=True):
            for key in edge_labels:
                edge[key] = random.uniform(0.3, 1.0)
        nodes_df = self.__create_note_df(G)
        edge_df = self.__create_edges_df(G, ['src', 'dst', 'weight', 'edge_lbl1', 'edge_lbl2'])
        self.data_feeder = DataFeederGraphFrames((nodes_df, edge_df), verbose=True)
        self.data_feeder_nx = DataFeederNx(G, verbose=True)

    # def test_init_train_batch(self):
    #     self.data_feeder.init_train_batch()
    #     self.assertIsInstance(self.data_feeder.iter["train"], tf.data.Dataset)

    def test_get_valid_node_labels(self):
        '''test to create is the feature numpy array is same as nx implementation
        '''
        features = self.data_feeder.features
        self.assertEqual(features.shape, (14,2))
        features_nx = self.data_feeder_nx.features
        np.testing.assert_array_equal(features, features_nx, 
            err_msg='feature frame differs from nx implementation'
            )

    def test_get_in_sample(self):
        ''' check if the in_sample is correct
        '''
        in_sample = self.data_feeder.in_sample
        in_sample_nx = self.data_feeder_nx.in_sample

        self.assertEqual(in_sample.shape, (14, 3))
        np.testing.assert_array_equal(in_sample, in_sample_nx, 
            err_msg='in sample frame differs from nx implementation'
            )

        self.assertListEqual(in_sample[1].tolist(), [2, 3, 0])
        self.assertListEqual(in_sample[7].tolist(), [8, 13, 13])
        self.assertListEqual(in_sample[13].tolist(), [13, 13, 13])


    def test_get_in_sample_weight(self):
        """ check if the in_sample edge features are correct
        """
        in_weight = self.data_feeder.in_sample_weight
        in_weight_nx = self.data_feeder_nx.in_sample_weight

        self.assertEqual(in_weight.shape, (14, 3, 3))
        dif = abs(sum(sum(sum(in_weight - in_weight_nx))))
        self.assertAlmostEqual(dif, 0.0, 6,
            msg='in edge feature sample frame differs from nx implementation'
        )

        self.assertEqual(in_weight.shape, (14, 3, 3))

        edge_ft = np.array(
            self.data_feeder.edge_df
            .filter("src = 2 and dst = 1")
            .select(self.data_feeder.edge_labels)
            .collect()
        )
        dif = self.data_feeder.in_sample_weight[1,0] - edge_ft
        self.assertAlmostEqual(np.sum(dif), 0.0, 6,
            msg='in edge feature [1,0] differs from expectation'
        )

    def test_get_out_sample(self):
        """ check if the out sample is correct
        """
        out_sample = self.data_feeder.out_sample
        out_sample_nx = self.data_feeder_nx.out_sample

        self.assertEqual(out_sample.shape, (14, 3))
        np.testing.assert_array_equal(out_sample, out_sample_nx, 
            err_msg='out sample frame differs from nx implementation'
            )

        self.assertListEqual(out_sample[1].tolist(), [2, 3, 0])
        self.assertListEqual(out_sample[7].tolist(), [6, 13, 13])
        self.assertListEqual(out_sample[6].tolist(), [13, 13, 13])
        self.assertListEqual(out_sample[13].tolist(), [13, 13, 13])

    def test_get_out_sample_weight(self):
        """ check if the outgoing edge features are correct
        """
        out_edge_feature = self.data_feeder.out_sample_weight
        out_edge_feature_nx = self.data_feeder_nx.out_sample_weight

        self.assertEqual(out_edge_feature.shape, (14, 3, 3))

        dif = abs(sum(sum(sum(out_edge_feature - out_edge_feature_nx))))
        self.assertAlmostEqual(dif, 0.0, 6,
            msg='out edge feature sample frame differs from nx implementation'
        )

        edge_ft = np.array(
            self.data_feeder.edge_df
            .filter("src = 1 and dst = 3")
            .select(self.data_feeder.edge_labels)
            .collect()
        )
        dif = self.data_feeder.out_sample_weight[1,1] - edge_ft
        self.assertAlmostEqual(np.sum(dif), 0.0, 6,
            msg='out edge feature from node 1 to second highest weight node 3 differs from expectation'
        )


    def test_feature_dim(self):
        """ checks if the nodes feature and total feature dim is correct
        """
        self.assertEqual(self.data_feeder.feature_dim, 2)
        self.assertEqual(self.data_feeder.get_feature_size(), 5)

    def test_train_batch(self):
        """ checks if the train batch is correctly initialized
        """
        self.data_feeder.init_train_batch()

        self.assertEqual(self.data_feeder.train_epoch_size, 9)
        self.assertEqual(self.data_feeder.val_epoch_size, 4)
        for i in self.data_feeder.iter['train'].take(1):
            self.assertEqual(i.shape, (3,))

    def test_dummy_id(self):
        """test if the correct dummy id is assigned to the dummy node
        """
        self.assertEqual(self.data_feeder.dummy_id, 13)

    def test_incr_iter(self):
        """ test if the incremental iterator has the correct size
        i.e. has the correct number of dummy nodes appended.
        """
        counter = 0
        for i in self.data_feeder.init_incr_batch():
            counter = counter + 1

        self.assertEqual(counter, 5)

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

        nodes_df = self.__create_note_df(graph)
        edge_df = self.__create_edges_df(graph, ['src', 'dst', 'weight', 'edge_lbl1'])
        data_feeders = [
            DataFeederGraphFrames((nodes_df, edge_df)),
            DataFeederGraphFrames((nodes_df, edge_df), weight_label='edge_lbl1')
        ]

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

    def __create_note_df(self, G):
        nodes = G.nodes(data=True)
        pdf = pd.DataFrame([[k] + list(v.values()) for k,v in nodes], columns= ['id', 'label1', 'label2'])
        return spark.createDataFrame(pdf)

    def __create_edges_df(self, G, lbls):
        edges = G.edges(data=True)
        pdf = pd.DataFrame([[s, d]+list(a.values()) for (s,d,a) in edges],
                            columns=lbls
        )
        return spark.createDataFrame(pdf)     

if __name__ == '__main__':
    unittest.main()