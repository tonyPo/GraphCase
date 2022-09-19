#%%
import networkx as nx
import pandas as pd
import numpy as np
import sys
ROOT_FOLDER = '/Users/tonpoppe/workspace/GraphCase'
sys.path.insert(0, ROOT_FOLDER)
from GAE.data_feeder_graphframes import spark
from  GAE.data_feeder_graphframes import DataFeederGraphFrames
from  GAE.data_feeder_nx import DataFeederNx
import examples.example_graph_bell as gb
import tensorflow as tf
import pyspark
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import random

#%%
# spark = SparkSession(sc)

G = gb.create_directed_barbell(4, 4)
random.seed(1)
edge_labels = ['weight', 'edge_lbl1', 'edge_lbl2']
for _, _, edge in G.edges(data=True):
    for key in edge_labels:
        edge[key] = random.uniform(0.3, 1.0)

nodes = G.nodes(data=True)
pdf = pd.DataFrame([[k] + list(v.values()) for k,v in nodes], columns= ['id', 'label1', 'label2'])
nodes_df = spark.createDataFrame(pdf)

edges = G.edges(data=True)
pdf = pd.DataFrame([[s, d]+list(a.values()) for (s,d,a) in edges],
                    columns=['src', 'dst'] + edge_labels
)
edges_df = spark.createDataFrame(pdf)
# %%
data_feeder = DataFeederGraphFrames((nodes_df, edges_df), verbose=True)
# gf_ft = data_feeder.features

# %%

data_feeder_nx = DataFeederNx(G, verbose=True)
# gf_nx = data_feeder_nx.features

#%%

in_edge = (
    nodes_df.select('id')
    .join(edges_df, nodes_df.id==edges_df.dst, 'left')
    .drop('dst')
)

in_edge = in_edge.withColumn('edge_features', F.array(edge_labels))

w = Window.partitionBy('id').orderBy(-F.col('weight')).rowsBetween(0, 2)
w2 = Window.partitionBy('id').orderBy(-F.col('weight'))
in_sample = (in_edge
    .withColumn('in_sample', F.collect_list('src').over(w))
    .withColumn('in_features', F.collect_list('edge_features').over(w))
    .withColumn('rn', F.row_number().over(w2))
    .filter("rn = 1")
    .select('id', 'in_sample', 'in_features')
    .withColumn('size', F.size(F.col('in_sample')))
    .withColumn('dummy_node', F.array_repeat(F.lit(0), 2 - F.col('size')))
    .withColumn('dummy_edge_features',  
                F.array_repeat(
                    F.array_repeat(F.lit(0), 
                        len(edge_labels)),
                    2 - F.col('size'))
                )
    .withColumn('in_sample', F.concat(F.col('in_sample'), F.col('dummy_node')))
    .withColumn('in_features', F.concat(F.col('in_features'), F.col('dummy_edge_features')))
)

#%%
def __create_note_df(self, G):
    nodes = G.nodes(data=True)
    pdf = pd.DataFrame([[k] + list(v.values()) for k,v in nodes], columns= ['id', 'label1', 'label2'])
    return spark.createDataFrame(pdf)

def __create_edges_df(self, G, lbls=['src', 'dst', 'weight', 'edge_lbl1', 'edge_lbl2']):
    edges = G.edges(data=True)
    pdf = pd.DataFrame([[s, d]+list(a.values()) for (s,d,a) in edges],
                        columns=[']
    )
    return spark.createDataFrame(pdf) 

#%%
graph = gb.create_directed_barbell(4, 4)
edge_weight = [1, 1, 1]
edge_lbl1_in = [0.22, 2.198239366963403, 3.1873590504451044]
edge_lbl1_out = [91.12909091, 0.71726504, 0.55211558]
for in_node, out_node, lbl in graph.edges(data=True):
    lbl['edge_lbl1'] = in_node/(out_node + 0.011) + 0.22


# %%
