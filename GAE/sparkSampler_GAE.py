
"""
Created on Tue Jul 23 07:33:36 2019

@author: tonpoppe
"""

# MAC OS bug
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# cwd = os.getcwd()
# os.chdir("/Users/tonpoppe/anaconda/lib/python3.5/site-packages")
# import findspark
# os.chdir(cwd)
# findspark.init()
# import pyspark
# import pyspark.sql
# from pyspark.sql import SparkSession
# from pyspark import SparkContext

# TensorFlow and tf.keras
# import tensorflow as tf


# Helper libraries
import pyspark.sql.functions as F
import pyspark.sql.types as t
from pyspark.sql.window import Window
import numpy as np
import pandas as pd
import math
import random


class SparkSampler:
    '''
    Takes the a feature file with one row per node and samples the first and
    second neighborhood of the node using the edges files. The samples size of
    the first and 2nd neighboorhood are parametized. This is done for matching
    couples of nodes and corresponding negative examples. The output is written
    to file per layer as one continuous file. The Samples can be used for the
    GraphSage Algo.
    Note that thefeatures of the second layer are already averages into one
    aggregated node.

    Additionally, it creates also an incremental file of all the node which can
    be used to create the embeddings of the complete network.

    Args:
      edges_file: hadoop file containing the edge information with the following
        naming convention; src, dst, in_scope. the in_scope parameter can be
        used to exclude edge when sampling.
      features_file, hadoop file containing the feature information with the
        following naming convention;id, in_scope. the in_scope parameter can be
        used to exclude nodes from the top layer in the batch, and the total file.
      out_dir: folder where the output files are written to.
      sep: the separator used for the input files
      batch_size: The number of node pairs in the minibatch.
      neg_sample_size: The number of nodes in the negative sample
      support_size: array with the support sizes per layer in de order from top
        layer to bottom layer. Note that the size of the array is used for
        determining the number of layers, usually only two.
      validation_fraction: fraction of nodes used for validation batch. The
        validation batch will have the same size of the train batches.

    '''

    def __init__(self,
                 edges_file,
                 features_file,
                 out_dir,
                 spark,
                 sep= ",",
                 support_size = [20,20],
                 validation_fraction = 0.1):
        """
        Initialisation of the class, seting the attributes
        """
        self.edges_file = edges_file
        self.features_file = features_file
        self.temp_dir = out_dir
        self.spark = spark
        self.e = None
        self.f = None
        self.sep = sep
        self.support_size = support_size
        self.validation_fraction = validation_fraction
        self.negative_sample_fraction = 0.3 #TODO this should be the ratio between 2 times the batch size and neg smaple
        self.dataset_size = None
        self.batchnr = 0
        self.all_L2_agg = None
        self.all_L1 = None

    def join_with_features(self, df, direction='out'):
        df = df.withColumnRenamed("id", "target_id")
        df = df.select("target_id", "adj", "amount").join(self.f, self.f.id == df.adj, 'inner')

        print("starting to aggregate ", direction)
        df = df.drop("id", "original_id", "adj", "in_scope", "is_val")
        group_labels = ["target_id"]
        features_labels = [x for x in df.columns if x not in group_labels]
        agg_features = [F.mean(x).alias(direction+"_"+x) for x in features_labels]
        df = df.groupBy(group_labels).agg(*agg_features)
        return df


    def create_nodes(self):
        print("creating node file")
        df = self.f
        excluded = ["original_id", "in_scope", "is_val", "id"]
        col_order = ["id"] + [c for c in self.f.columns if c not in excluded]
        file_name = self.temp_dir+"/"+"node_labels"
        df.select(col_order).coalesce(10).write.csv(file_name, mode='overwrite', header=True, compression='gzip')

    def create_in_and_out_sample(self):
        print("Start creation of layer 1 samples")
        df_out = self.retrieve_next_layer(self.f, self.e, self.support_size[0])
        df_out.persist()

        # create out sample id
        out_layer = df_out.groupBy("id").pivot('rank').agg(F.first(F.col('adj')))
        dummy_id = self.f.filter("original_id = 'abnanl_dummy0001'").select("id").collect()[0][0]
        out_layer = out_layer.na.fill(str(dummy_id))
        out_layer.coalesce(10).write.csv(self.temp_dir+"/out_sample", mode='overwrite', header=True, compression='gzip')

        # create out sample amount
        out_layer = df_out.groupBy("id").pivot('rank').agg(F.first(F.col('amount')))
        out_layer = out_layer.na.fill(0)
        out_layer.coalesce(10).write.csv(self.temp_dir+"/out_sample_amnt", mode='overwrite', header=True, compression='gzip')

        # create out sample id
        df_in = self.retrieve_next_layer(self.f, self.e, self.support_size[0], direction = 'in')
        df_in.persist()
        in_layer = df_in.groupBy("id").pivot('rank').agg(F.first(F.col('adj')))
        in_layer = in_layer.na.fill(str(dummy_id))
        in_layer.coalesce(10).write.csv(self.temp_dir+"/in_sample", mode='overwrite', header=True, compression='gzip')

        # create in sample amount
        in_layer = df_in.groupBy("id").pivot('rank').agg(F.first(F.col('amount')))
        in_layer = in_layer.na.fill(0)
        in_layer.coalesce(10).write.csv(self.temp_dir+"/in_sample_amnt", mode='overwrite', header=True, compression='gzip')


    def retrieve_next_layer(self, f, e, topx, direction='out'):
        if direction == 'out':
            orig_node = 'src'
            dest_node = 'dst'
        else:
            orig_node = 'dst'
            dest_node = 'src'

        df = f.select("id").join(e.drop('in_scope'), f.id == e[orig_node], 'inner').drop(orig_node)
        window = Window.partitionBy(df['id']).orderBy(df['amount'].desc())
        df = df.select('*', F.rank().over(window).alias('rank')).filter(F.col('rank') <= topx)

        dummy_tmp = self.create_dummy_edges(df, f, topx, direction)
        df = dummy_tmp.union(df.select(dummy_tmp.columns))
        df = df.withColumn("direction", F.lit(direction))
        df = df.withColumnRenamed(dest_node, "adj")

        return df

    def create_dummy_edges(self, df, f, topx, direction='out'):
        if direction == 'out':
            dest_node = 'dst'
        else:
            dest_node = 'src'

            # group per id and count edges

        dummy_id = int(self.f.filter("original_id = 'abnanl_dummy0001'").select("id").collect()[0][0])
        tmp = df.groupBy("id").count()
        print("number of accounts after groupby : ", tmp.count())
        # join to original feature frame to include features without edges
        tmp = tmp.join(f.select("id"), 'id', 'right')
        tmp = tmp.fillna(0)
        print("number of accounts after join and groupby : ", tmp.count())

        # create function to return list 8times
        def create_list_dummy(count):
            if count > 0:
                res = [dummy_id] * count
            else:
                res = []

            return res

        uniform_sample_udf = F.udf(create_list_dummy, t.ArrayType(t.StringType()))

        #determine the number dummies to be created
        df_dummy = tmp.withColumn("dummy_cnt", F.lit(topx) - F.col("count"))
        df_dummy = df_dummy.filter("dummy_cnt > 0")

        #create dummies
        df_dummy = df_dummy.withColumn("dummy_list", uniform_sample_udf(F.col("dummy_cnt")))
        df_dummy = df_dummy.select(["id", F.explode(df_dummy["dummy_list"]).alias(dest_node)])
        df_dummy = df_dummy.withColumn("amount", F.lit(0))
        df_dummy = df_dummy.withColumn("rank", F.lit(topx))

        print("number of dummys created: ", df_dummy.count())
        # print(df_dummy.collect())
        return df_dummy

    def create_sample(self, only_incremental = False):
        """
        Creates the samples for the train, validation and incremental batch,
        which contains all nodes where the in_scope field set to true.
        The output is saves to separate files per layer and type.
        """
        # load data
        self.load_and_prep_data()
        self.create_in_and_out_sample()
        self.create_nodes()

        # create train and validation set
        if not only_incremental:
            for tp in ["train","valid"]:
                is_val = 0 if  tp == "valid" else 1
                self.create_l0_batch(tp, is_val)

        #create incremental batch
        self.create_incremental_batch()



    def create_dummy_node(self, f):
        """
        Creates dummy node with all attributes set to zero
        The dummy id is abnanl_dummy0001
        :param f: feature dataframe from which the columns names are copied
        :return: dataframe with one dummy row whos values are set to zero
        """
        feature_names = [f for f in f.columns if f not in ['id']]
        values = [0]*len(feature_names)
        cSchema = t.StructType([t.StructField(f, t.IntegerType()) for f in feature_names])

        dummy_df = self.spark.createDataFrame([values], cSchema)
        dummy_df = dummy_df.withColumn("id", F.lit("abnanl_dummy0001"))
        return dummy_df


    def load_and_prep_data(self):
        """
        load the edge and feature data
        sets defined fraction of the edge which are in scope to validation
        sets the nodes in validation edge to validation
        """
        self.e = self.spark.read.csv(self.edges_file, sep = self.sep, header=True)
        self.e = self.e.withColumn("amount", F.col('amount').cast('float'))
        print("count for loaded edges: " + str(self.e.count()))

        self.f = self.spark.read.csv(self.features_file, sep = self.sep, header=True)
        # create dummy node
        dummy_node = self.create_dummy_node(self.f)
        self.f = self.f.union(dummy_node.select(self.f.columns))

        # set validation nodes
        self.f = self.f.withColumn("is_val",
                                   F.when(F.col("in_scope") == 0, F.lit(0)).otherwise(
                                       F.array(F.lit(0),F.lit(1)).getItem(
                                           F.when(F.lit(F.rand()) > self.validation_fraction, 0).otherwise(1)
                                       )))
        self.f.persist()
        self.f = self.f.withColumnRenamed("id","original_id")

        print(" features in scope " + str(self.f.filter("in_scope=1").count()))
        print(" features for validation " + str(self.f.filter("in_scope=1 and is_val = 1").count()))
        print(" all features " + str(self.f.count()))
        self.f = self.add_incremental_id(self.f)
        self.f.persist()
        self.reset_edge_ids()

    def reset_edge_ids(self):
        df = self.f.select(['id', 'original_id']).join(self.e, self.f.original_id == self.e.src, 'inner')
        df = df.drop('original_id', 'src')
        df = df.withColumnRenamed('id', 'src')
        df = df.join(self.f.select(['id', 'original_id']), self.f.original_id == df.dst, 'inner')
        df = df.drop('original_id', 'dst')
        df = df.withColumnRenamed('id', 'dst')
        self.e = df
        self.e.persist()

    def create_l0_batch(self, name, is_val=0):
        """
        Creates the L0 layer of a batch
        input
            is_val:  indicator in range {0,1} whether a train of validation
                batch needs to be created
        """
        df = self.f.filter("in_scope = 1 and is_val = " + str(is_val)).select('id')
        df.coalesce(1).write.csv(self.temp_dir + "/" + name, mode='overwrite', header=True)

    def create_incremental_batch(self):
        """
        Creates and returns a dataframe with all nodes for with the embedding needs to be calculated
        """
        inc = self.f.filter(F.col("in_scope") == 1).select("id")
        inc.persist()
        inc_cnt = inc.count()
        # add batch with a multiple of 1024
        if inc_cnt%1024 != 0:
            padding_size = 1024 - inc_cnt%1024
            inc_add = self.f.select("id").limit(padding_size)
            inc = inc.union(inc_add)

        inc.coalesce(1). \
                write.csv(self.temp_dir + "/incr", mode='overwrite', header=True)

    def add_incremental_id(self , f):
        pd_df = f.select("original_id").toPandas()

        pd_df = pd_df.reset_index()
        # pd_df.columns[0] = 'nid'
        pd_df['id'] = pd_df.index

        print( "rows in pandas dataframe ", len(pd_df.index))
        pd_df.to_csv("/dbfs" + self.temp_dir + "index_map")

        cSchema = t.StructType([t.StructField("id", t.IntegerType()),
                                t.StructField("original_id", t.StringType())])

        df = self.spark.createDataFrame(pd_df[["id","original_id"]],schema=cSchema)

        f = f.join(df, 'original_id', 'inner')
        return f
