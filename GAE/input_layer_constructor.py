#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21-06-2021

@author: tonpoppe
"""

import tensorflow as tf
from GAE.data_feeder_nx import DataFeederNx

class InputLayerConstructor:
    """
    Samples the graph in a deterministic based on edge weight
       
        features:   a 2d numpy dataframe containing the feature information.
        in_sample:  a 2d numpy matrix with the node ids of the incoming neighbourhood.
        out_sample: a 2d numpy matrix with the node ids of the outgoing neighbourhood.
        in_sample_weight: a 3d numpy matrix with the sample weight of the incoming
                    neighbourhood. (batch_size, support_size, len(edge_labels))
        out_sample_weight: a 2d numpy matrix with the sample weight of the outgoing
                    neighbourhood. (batch_size, support_size, len(edge_labels))
    """
    def __init__(self, graph, support_size, batch_size=3, val_fraction=0.3,
                 verbose=False, seed=1, weight_label='weight', encoder_labels=None):
        self.data_feeder = DataFeederNx(
            graph, neighb_size=max(support_size),batch_size=batch_size, verbose=verbose, seed=seed,
            weight_label=weight_label, val_fraction=val_fraction, encoder_labels=encoder_labels)
        self.support_size = support_size
        self.features = tf.constant(self.data_feeder.features, name="features")
        self.in_sample = tf.constant(self.data_feeder.in_sample, dtype=tf.int64, name="in_sample")
        self.out_sample = tf.constant(self.data_feeder.out_sample, dtype=tf.int64, name="out_sample")
        self.in_sample_amnt = tf.constant(self.data_feeder.in_sample_weight, name="in_sample_amnt")
        self.out_sample_amnt = tf.constant(self.data_feeder.out_sample_weight, name="out_sample_amnt")

    def init_train_batch(self, label_name=None):
        self.data_feeder.init_train_batch(label_name=label_name)

    def get_epoch_sizes(self):
        return (self.data_feeder.train_epoch_size, self.data_feeder.val_epoch_size)

    def get_feature_size(self):
        """
        returns the number of node labels + edge labels
        """
        return self.data_feeder.get_feature_size()

    def get_number_of_node_labels(self):
        """
        Return the number of labels that a node has, excl edge labels
        """
        return self.data_feeder.get_number_of_node_labels()

    def init_incr_batch(self, nodes):
        """
        returns a dataset with tuple(node_ids, node_features, neighbourhood)
        """
        batch = self.data_feeder.init_incr_batch(nodes)
        return batch.map(lambda x: (x, self.get_features(x), self.get_input_layer(x, hub=1)[0]))

    def get_train_samples(self):
        train_data = self.data_feeder.get_train_samples()
        train_data = train_data.map(lambda x: (self.get_features(x), self.get_input_layer(x, hub=1)))
        return train_data.map(lambda x, i: ((x, i[0]), (x, i[0]), (1, i[1])))

    def get_supervised_train_samples(self):
        return self.__get_supervised_samples(self.data_feeder.get_train_samples)

    def get_supervised_val_samples(self):
        return self.__get_supervised_samples(self.data_feeder.get_val_samples)

    def __get_supervised_samples(self, feeder):
        data = feeder()
        data = data.map(lambda x: (x, tf.nn.embedding_lookup(self.data_feeder.lbls, x)))
        data = data.map(lambda x, y: (self.get_features(x), self.get_input_layer(x, hub=1), y))
        return data.map(lambda x, i, y: ((x, i[0]), y, (1, i[1])))

    def get_val_samples(self):
        val_data = self.data_feeder.get_val_samples()
        val_data = val_data.map(lambda x: (self.get_features(x), self.get_input_layer(x, hub=1)))
        return val_data.map(lambda x, i: ((x, i[0]), (x, i[0]), (1, i[1])))


    def get_features(self, batch):
        return tf.nn.embedding_lookup(self.features, batch)

    def get_input_layer(self, node_ids, hub, feat=None, weight=None):
        """
        Retrieve the first input layer by sampling the graph. This method is called
        recursively per hub.

        Args:
            node_ids:   tensor with the node ids for which the input_layer needs to be calculated.
            hub:        hub of the inputlayer, for hub = 1.
            feat:       In case this is called recursively, a tensor with the features of the lower
                        hub which will be combined in this hub.
            weight:     In case of a recursive call, a tensor with the weights of the lower hubs
                        which will be used for multiplication of this hubs weights.

        returns:
            a tuple with 1) a tensor of the features for the specified hub and 2) a tensor with
            the weights for the specified hub.
        """
        next_in_feat, next_in_weight = self.__get_next_hub(node_ids, hub, 'in', feat, weight)
        next_out_feat, next_out_weight = self.__get_next_hub(node_ids, hub, 'out', feat, weight)

        if feat is not None:
            if hub == len(self.support_size):
                feat = tf.concat([feat, next_in_feat, feat, next_out_feat], -2)

                shape = tf.shape(weight)
                head = None
                # for level in range(shape[-1]):
                for level in range(hub - 1):
                    factor = tf.slice(weight, 
                        tf.concat([tf.repeat([0], tf.shape(shape)-1), [level]], axis=0),
                        tf.concat([shape[:-1], [1]], axis=0))
                    next_in_weight = tf.math.multiply(next_in_weight, factor)
                    next_out_weight = tf.math.multiply(next_out_weight, factor)
                    if head is None:
                        head = tf.math.pow(factor, tf.cast(shape[-1] - level + 1, tf.float32))
                        tail = tf.slice(weight,
                            tf.concat([tf.repeat([0], tf.shape(shape)-1), [level+1]], axis=0),
                            tf.concat([shape[:-1], [shape[-1] - 1]], axis=0))
                        tail = tf.math.multiply(tail, factor)
                    else:
                        tail = tf.math.multiply(tail, factor)
                        head_add = tf.slice(tail,
                            tf.concat([tf.repeat([0], tf.shape(shape)-1), [0]], axis=0),
                            tf.concat([shape[:-1], [1]], axis=0))
                        head = tf.concat([head, head_add], -1)
                        tail = tf.slice(tail,
                            tf.concat([tf.repeat([0], tf.shape(shape)-1), [1]], axis=0),
                            tf.concat([shape[:-1], [shape[-1] - 1 - level]], axis=0))

                weight_comb = tf.concat([head, next_in_weight, head, next_out_weight], -1)

            else:
                feat = tf.concat([next_in_feat, next_out_feat], -2)
                weight_comb = tf.concat([next_in_weight, next_out_weight], -1)

            shape = tf.shape(feat)
            new_shape = tf.concat([shape[:-3], [shape[-3] * shape[-2], shape[-1]]], axis=0)
            feat = tf.reshape(feat, new_shape)
            weight = tf.reshape(weight_comb, new_shape[:-1])

        else:
            feat = tf.concat([next_in_feat, next_out_feat], -2)
            weight = tf.concat([next_in_weight, next_out_weight], -1)

        return feat, weight

    def __get_next_hub(self, node_ids, hub, direction, feat, weight):
        """
        Retrieves the features including edge weights for the specified hub and direction.
        The features are combined with the features of the lower hubs.

        Args:
            node_ids:   A tensor containing the node_ids for which the featurs need to be
                        retrieved.
            hub:        integer specifying the hub for which the features need to be retrieved.
            direction:  {in, out} indicator whether the features for the incoming or outgoing
                        neighbourhood need to be retrieved.
            feat:       A tensor with the features of the lower
                        hub which will be combined in this hub.
            weight:     A tensor with the weights of the lower hubs
                        which will be used for multiplication of this hubs weights.

        Returns:
            A tuple of two tensors with 1) a tensor with the features and 2 a tensor with the
            corresponding weights.

        """
        support = self.support_size[hub-1]
        if direction == 'in':
            sample_node = self.in_sample[:, :support]
            sample_weight = self.in_sample_amnt[:, :support]
        else:
            sample_node = self.out_sample[:, :support]
            sample_weight = self.out_sample_amnt[:, :support]

        next_nodes = tf.nn.embedding_lookup(sample_node, node_ids)
        weight_next = tf.nn.embedding_lookup(sample_weight, node_ids)
        feat_next = tf.nn.embedding_lookup(self.features, next_nodes)
        #combine feature + edge labels
        feat_next = tf.concat([weight_next, feat_next], -1)
        weight_next = tf.slice(weight_next, tf.repeat([0], tf.shape(tf.shape(weight_next))),
                               tf.concat([tf.shape(weight_next)[:-1], [1]], axis=0))
        weight_next = tf.squeeze(weight_next)
        # check weight_next = weight_next[...,0]

        if hub < len(self.support_size):
            # add additional dimension for the next hub features and weight
            feat_next = tf.expand_dims(feat_next, -2)
            weight_next = tf.expand_dims(weight_next, -1)
            if feat is not None:
                shape = tf.shape(feat)
                # tile_shape = [1] * (len(shape) -2) + [self.support_size[hub - 1]] + [1]
                tile_shape = tf.concat([
                    tf.repeat([1], tf.shape(shape)-2),
                    [self.support_size[hub - 1]],
                    [1]
                    ], axis=0
                )
                feat = tf.tile(feat, tile_shape)
                feat = tf.expand_dims(feat, -2)
                feat_next = tf.concat([feat, feat_next], -2)
                weight = tf.tile(weight, tile_shape[:-1])
                weight = tf.expand_dims(weight, -1)
                weight_next = tf.concat([weight, weight_next], -1)


            feat_next, weight_next = self.get_input_layer(next_nodes, hub+1,
                                                            feat_next, weight_next)

        return feat_next, weight_next