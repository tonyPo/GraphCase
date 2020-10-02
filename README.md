# GraphCase
Inductive graph auto encode for creating embeddings on weighted, directed graphs including node and edge label information. The algorithm is based on a **CA**nonical ordering of the graph neighbourhoud based on the edge weight or properties combined with either a stochastic or deterministic **S**ampling approach. The sampled neightbourbour is fed into a standard auto-**E**ncoder to create an embeding per layer. Hence the name GraphCase: **CA**nonical ordering and **S**ampling **E**ncoder.  By iteration over multiple layer the scope of the resulting embedding is increased. 

## context
Various classification and clustering problems benefit if the structural identity and properties of a node in a graph are combined. For example when trying to detect money laundering, which aims at hiding the illicit origin of the money via a series of transactions. Financial transactions between entities can be represented as a graph in which sending or receiving parties are represented as a node and transactions as an edge. The transactions which are part of the money laundering scheme can be represented as a subgraph of the transaction graph. However, the money laundering schemes can't be detected on solely the structure of the subgraph. Important label information is required like the country of the entities and the transferred amounts. Vice versa, it is very hard to detect money laundering solely on label information of a single customer as this give a too narrow view.

GraphCASE aims at creating representations that combine both the label information of the nodes and edges as well as the local graph neighbourhood. These representations can be used for classification, clustering or regression purposes. The algorithm is inspired by the GraphSAGE algorithm of Hamilton[1] with a number of important changes. The algorithm first samples the local neighbourhood in a canonical order. The order of sampling is based on the edge weight and every node can only be sampled once. This ensures that the most informative nodes are included in the sampled neighbourhood and that the sample has a clear distinction between nodes with only has one neighbour (which would be repetitively sampled in GraphSAGE) versus node with multiple similar neighbours. To ensure that the size of the sample is uniform for every node we introduce dummy nodes and fill the sample to a predefined size with dummy nodes. The labels of the sampled nodes together with the edge labels are ordered in a canonical order with a repetitive structure. Note that this contains all relevant information that we want to capture in the node representation. We use a series of "convolutional" like auto-encoders to convert this ordered set of node and edge labels into a representation.

## Motivation for GraphCASE
Note that the canonical ordering based on the edge weight allows for emphasizing on those nodes and edges that are important for the job at hand. For example in the case of money laundering detection, only the transactions with a significant amount are normally part of the money laundering scheme, which is often only a small subset of the total number of transactions. Or in the case of social network analysis such as LinkedIn, one would like to include the strength of the relationship when creating the representation.

Adding the edge labels and direction of the edge allows the representation to capture the type of relation between the nodes. For some use case this can contain valuable information. For example in the case of identifying a money laundering scheme, the direction of the transaction contains important information. Additionally, it allows for capturing characteristics of the transactions. Similar in the case of social network analysis, the type of connection between two persons contains relevant information, i.e. are the colleagues, friends, family, etc. Note that Kipf[2] states the this information can be captured in a undirected graph by converting the edges into additional nodes. But this pushes the relevant information further into the subgraph and might therefore not be captured as good as would be without this adjustment.

## example
Below a toy example is shown. The graph on the right show a directed graph consisting of the fully connected cliques of size 10 connected via a path of size 10. the nodes has two labels (refer to wiki for details) and the egde have only one weight label. The edge color indicates the edge weight.

![Alt text](https://github.com/tonyPo/GraphCase/blob/feature/experiments/graphbell.png?raw=true "Graph bell" =150x)

[1]:  William L Hamilton.  Inductive Representation Learning on Large Graphs.(Nips):1–11, 2017.  
[2]:  Thomas  N  Kipf  and  Max  Welling.    Semi-Supervised  Classification  withGraph Convolutional Networks.  pages 1–14, 2016.
