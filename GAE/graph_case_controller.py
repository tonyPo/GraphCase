#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 07:26:20 2019

@author: tonpoppe
"""
import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from GAE.model import GraphAutoEncoderModel
from GAE.input_layer_constructor import InputLayerConstructor
from GAE.graph_reconstructor import GraphReconstructor
from GAE.transformation_layer import DecTransLayer, EncTransLayer, Hub0_encoder, Hub0_decoder
from GAE.data_feeder_nx import DataFeederNx
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
class GraphAutoEncoder:
    """
    This class implement the graphCase algorithm. Refer for more details
    to the corresponding documentation.

    Args:
        graph:      graph on which the embedding is trained. Only bi-directed
                    graphs re supported.
        learning_rate: learning rate of the MLP.
        support_size: list with number of sampled edges per layer. The
                    current implementation only support one size for all layers
        dims:       list with the dimension per layer.
        batch_size: number of nodes per training cycle.
        max_total_steps: Number of batches used for training the mlp.
        validate_iter: Number of batches between a validation batch.
        verbose:    boolean if True then detailed feedback on the training progress
                    is given.
        seed:       Seed used for the random split in train and test set.

    """
    def __init__(self,
                 graph=None,
                 learning_rate=0.0001,
                 support_size=[2, 2],
                 dims=[32, 32, 32, 32],
                 hub0_feature_with_neighb_dim=None,
                 batch_size=3,
                 verbose=False,
                 seed=1,
                 weight_label='weight',
                 encoder_labels=None,
                 act=tf.nn.sigmoid,
                 useBN=False,
                 val_fraction=0.3,
                 model_config=None,
                 dropout=False,
                 data_feeder_cls=DataFeederNx
                 ):
        self.learning_rate = learning_rate
        self.dims = dims
        self.hub0_feature_with_neighb_dim = hub0_feature_with_neighb_dim
        self.batch_size = batch_size
        self.support_size = support_size
        self.verbose = verbose
        self.seed = seed
        self.act = act
        self.weight_label = weight_label
        self.encoder_labels = encoder_labels
        self.useBN = useBN
        self.dropout = dropout
        self.val_fraction = val_fraction
        self.data_feeder_cls = data_feeder_cls
        if graph is not None:
            self.__consistency_checks()
            self.sampler = self.__init_sampler(graph, val_fraction)
            self.model = self.__init_model()
        if model_config is not None:
            custom_objects = {
                "DecTransLayer": DecTransLayer,
                "EncTransLayer": EncTransLayer,
                "Hub0_encoder": Hub0_encoder,
                " Hub0_decoder": Hub0_decoder
            }
            with tf.keras.utils.custom_object_scope(custom_objects):
                self.model = GraphAutoEncoderModel.from_config(model_config)

    def __init_sampler(self, graph, val_fraction):
        """
        Initialises the datafeeder
        """
        return InputLayerConstructor(
            graph, support_size=self.support_size, val_fraction=val_fraction,
            batch_size=self.batch_size, verbose=self.verbose, seed=self.seed,
            weight_label=self.weight_label, encoder_labels=self.encoder_labels,
            data_feeder_cls=self.data_feeder_cls
        )

    def __init_model(self):
        """
        Initialises the model
        """
        model = GraphAutoEncoderModel(
            self.dims, self.support_size, self.sampler.get_feature_size(),
            hub0_feature_with_neighb_dim=self.hub0_feature_with_neighb_dim,
            number_of_node_labels=self.sampler.get_number_of_node_labels(),
            verbose=self.verbose, seed=self.seed, dropout=self.dropout, act=self.act,
            useBN=self.useBN)

        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.optimizers.RMSprop(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        self.sampler.init_train_batch()
        train_data = self.sampler.get_train_samples()
        for n in train_data.take(1):
            model(n[0])
        return model

    def calculate_embeddings(self, graph=None, nodes=None, verbose=False):
        """
        Calculated the embedding of the nodes specified. If no nodes are
        specified, then the embedding for all nodes are calculated.

        Args:
            graph:  Optionally the graph for which the embeddings need to be calculated. If set to
                    None then the graph used for initializing is used.
            nodes:  Optionally a list of node ids in the graph for which the
                    embedding needs to be calculated.

        Returns:
            A 2d numpy array with one embedding per row.
        """
        self.verbose = verbose
        if verbose:
            print("calculating all embeddings")

        if graph is not None:
            self.sampler = self.__init_sampler(graph, self.val_fraction)

        embedding = None
        counter = 0
        for i in self.sampler.init_incr_batch(nodes):
            counter += 1
            try:
                embed = self.model.calculate_embedding(i)
                if embedding is None:
                    embedding = embed
                else:
                    embedding = np.vstack([embedding, embed])

                if counter % 100 == 0:
                    print("processed ", counter, " batches time: ", datetime.now())

            except tf.errors.OutOfRangeError:
                break

        if verbose:
            print("reached end of batch")
        return embedding

    def save_model(self, save_path):
        """
        Saves the model. Note that a reloaded model can only be called.

        Args:
            save_path: path in which the model is stored.
        """
        self.model.save(save_path)
        attr_dict = {
            "learning_rate": self.learning_rate,
            "dims": self.dims,
            "hub0_feature_with_neighb_dim": self.hub0_feature_with_neighb_dim,
            "batch_size": self.batch_size,
            "support_size": self.support_size,
            "verbose": self.verbose,
            "seed": self.seed,
            "act": self.act,
            "weight_label": self.weight_label,
            "useBN": self.useBN,
            "val_fraction": self.val_fraction,
            # "model_config": self.model.get_config()
        }
        pickle.dump(attr_dict, open(f"{save_path}/params.pickle", "wb"))

    @classmethod
    def load_model(cls, save_path):
        """
        Loads a trained model from a pickle file.
        Note that the restored model can only be called.

        Args:
            filename: path with the stored model.
        """
        params = pickle.load(open(f"{save_path}/params.pickle", "rb"))
        new_gae = cls(graph=None, **params)
        new_gae.model = tf.keras.models.load_model("saved_model")
        return new_gae

    def save_weights(self, save_path):
        """
        Saves the weight of the model. These weight can be used to reconstructed the model for
        those cases where the model will be updated or changed

        Args:
            save_path:  The path where the weights are saved to.
        """
        self.model.save_weights(save_path)

    def load_weights(self, save_path):
        """
        Loads earlier saved weights back into the model. Note that we assume that the model has
        the same configuration as the model of the saved weights

        Args:
            save_path:  The folder containing the saved weights.
        """
        self.model.load_weights(save_path)

    def __consistency_checks(self):
        """
        Performs the following consistency checks.
        1) len of dims list is 2 * len support size or len is 2 * support size + 1
        """
        assert len(self.dims) == 2 * len(self.support_size) or \
               len(self.dims) -1 == 2 * len(self.support_size), \
               f"number of dims {len(self.dims)} does not match with two times the number of " \
               f"support sizes {len(self.support_size)}"

    def fit(self, graph=None, verbose=None, layer_wise=False, epochs=4):
        """
        Trains the model.

        Args:
            graph:  The graph used for training. If None then the graph for initializing the model
                    is used.
            verbose: Boolean to indicate whether information during training needs to be shown.
            layer_wise: Boolean to indicate whether the model needs to trained layer by layer or
                        all at once.
            epochs: Number of epochs used for training.

        Returns:
            Dict with the training results.
        """
        hist = {}
        if verbose is not None:
            self.verbose = verbose
        model_verbose = 1 if self.verbose else 0

        if graph is not None:
            self.sampler = self.__init_sampler(graph, self.val_fraction)

        layers = [None]
        if layer_wise:
            layers = [i for i in range(len(self.dims))] + layers

        for _, l in enumerate(layers):
            self.model.sub_model_layer = l
            self.sampler.init_train_batch()
            train_data = self.sampler.get_train_samples()
            validation_data = self.sampler.get_val_samples()

            train_epoch_size, val_epoch_size = self.sampler.get_epoch_sizes()
            steps_per_epoch = int(train_epoch_size / self.batch_size)
            assert steps_per_epoch>0, "batch_size greater then 1 train epoch"
            validation_steps = int(val_epoch_size / self.batch_size)
            assert validation_steps>0, "batch_size greater then 1 validation epoch"
            # early_stop = tf.keras.callbacks.EarlyStopping(
            #     monitor='val_loss', min_delta=0, patience=3, verbose=0
            # )
            history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                verbose=model_verbose,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                # callbacks=[early_stop]
            )
            hist[l] = history
        return hist

    def fit_supervised(
        self, label_name, model, compiler_dict, train_dict, graph=None, verbose=None):
        """
        expends the GAE with a supervised model and trains the model and the given graph or if
        none is provide the current graph.

        Args:
            label_name: The name of the node label containing the label information.
            model:      The supervised part of the model. The output of the encoder is fed into the
                        supervised part.
            compiler_dict: Dict with the parameter to be used for compiling the model.
            train_dict: Dict with the training parameter to be used for training the model.
            graph       The graph on which the model will be trained.

        """
        if verbose is not None:
            self.verbose = verbose
        model_verbose = 1 if self.verbose else 0

        if graph is not None:
            self.sampler = self.__init_sampler(graph, self.val_fraction)

        self.model.create_supervised_model(model)
        self.model.compile(**compiler_dict)
        
        self.sampler.init_train_batch(label_name)
        train_data = self.sampler.get_supervised_train_samples()
        validation_data = self.sampler.get_supervised_val_samples()

        train_epoch_size, val_epoch_size = self.sampler.get_epoch_sizes()
        steps_per_epoch = int(train_epoch_size / self.batch_size)
        validation_steps = int(val_epoch_size / self.batch_size)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=3, verbose=0
        )
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            verbose=model_verbose,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stop],
            **train_dict
        )
        return history


    def get_l1_structure(self, node_id, graph=None, verbose=None, show_graph=False,
                         node_label=None, get_pyvis=False, deduplicate=True,
                         delta=0.0001, dummy=0):
        """
        Retrieve the input layer and corresponding sampled graph of the local neighbourhood.

        Args:
            node_id:    id of the node for which the input layer is calculated
            graph:      graph used for sampling. If no graph is specified then the current graph
                        is used.
            show_graph  Boolean indicating if a plot of the graph needs to be generated.
            node_label  Label used for plotting the nodes. If None then the node id is used.

        returns:
            a networkx graph of the sampled neighbourhood and a numpy matrix of the input layer.
        """
        if verbose is not None:
            self.verbose = verbose

        if graph is not None:
            self.sampler = self.__init_sampler(graph, self.val_fraction)

        inputlayer, _ = self.sampler.get_input_layer([node_id], hub=1)
        target = self.sampler.get_features(node_id)
        graph_rec = GraphReconstructor(deduplicate=deduplicate, delta=delta, dummy=dummy)
        recon_graph = graph_rec.reconstruct_graph(target, inputlayer, self.support_size)

        if show_graph:
            graph_rec.show_graph(recon_graph, node_label=node_label)

        if get_pyvis:
            nt = graph_rec.show_pyvis(recon_graph, node_label=node_label)
            return inputlayer, recon_graph, nt

        return inputlayer, recon_graph

    def decode(self, embedding, incl_graph=None, delta=0.0001, dummy=0, deduplicate=True):
        """
        Decodes the given embedding into a node and local neighbourhood.
        Args:
            embedding:  Embedding of the node
            incl_graph :{None | pyvis | graph }
            delta:      Min difference between reconstructed feature value and dummy value. Nodes
                        with a smaller difference are considered dummy nodes and are removed.
            dummy:      Value of the dummy node.

        Returns:
            A tuple with the node labels, inputlayer and optionally a graph.
        """
        feat_out, df_out = self.model.decode(embedding)
        if incl_graph is not None:
            graph_rec = GraphReconstructor(delta=delta, dummy=dummy, deduplicate=deduplicate)
            recon_graph = graph_rec.reconstruct_graph(feat_out, df_out, self.support_size)

        if incl_graph == 'graph':
            return feat_out, df_out, recon_graph

        if incl_graph == 'pyvis':
            nt = graph_rec.show_pyvis(recon_graph)
            return feat_out, df_out, nt

        return feat_out, df_out, None
