# GraphCase
Inductive graph auto encode for creating embeddings on weighted, directed graphs including node and edge label information. The algorithm is based on a **CA**nonical ordering of the graph neighbourhoud based on the edge weight or properties combined with either a stochastic or deterministic **S**ampling approach. The sampled neightbourbour is fed into a standard auto-**E**ncoder to create an embeding per layer. Hence the name GraphCase: **CA**nonical ordering and **S**ampling **E**ncoder.  By iteration over multiple layer the scope of the resulting embedding is increased. 

## context
Various classification and clustering problems benefit if the structural identity and properties of a node in a graph are combined. For example when trying to detect money laundering, which aims at hiding the illicit origin of the money via a series of transactions. Financial transactions between entities can be represented as a graph in which sending or receiving parties are represented as a node and transactions as an edge. The transactions which are part of the money laundering scheme can be represented as a subgraph of the transaction graph. However, the money laundering schemes can't be detected on solely the structure of the subgraph. Important label information is required like the country of the entities and the transferred amounts. Vice versa, it is very hard to detect money laundering solely on label information of a single customer as this give a too narrow view.

GraphCASE aims at creating representations that combine both the label information of the nodes and edges as well as the local graph neighbourhood. These representations can be used for classification, clustering or regression purposes. The algorithm is inspired by the GraphSAGE algorithm of Hamilton[1] with a number of important changes. The algorithm first samples the local neighbourhood in a canonical order. The order of sampling is based on the edge weight and every node can only be sampled once. This ensures that the most informative nodes are included in the sampled neighbourhood and that the sample has a clear distinction between nodes with only has one neighbour (which would be repetitively sampled in GraphSAGE) versus node with multiple similar neighbours. To ensure that the size of the sample is uniform for every node we introduce dummy nodes and fill the sample to a predefined size with dummy nodes. The labels of the sampled nodes together with the edge labels are ordered in a canonical order with a repetitive structure. Note that this contains all relevant information that we want to capture in the node representation. We use a series of "convolutional" like auto-encoders to convert this ordered set of node and edge labels into a representation.

## Motivation for GraphCASE
Note that the canonical ordering based on the edge weight allows for emphasizing on those nodes and edges that are important for the job at hand. For example in the case of money laundering detection, only the transactions with a significant amount are normally part of the money laundering scheme, which is often only a small subset of the total number of transactions. Or in the case of social network analysis such as LinkedIn, one would like to include the strength of the relationship when creating the representation.

Adding the edge labels and direction of the edge allows the representation to capture the type of relation between the nodes. For some use case this can contain valuable information. For example in the case of identifying a money laundering scheme, the direction of the transaction contains important information. Additionally, it allows for capturing characteristics of the transactions. Similar in the case of social network analysis, the type of connection between two persons contains relevant information, i.e. are the colleagues, friends, family, etc. Note that Kipf[2] states the this information can be captured in a undirected graph by converting the edges into additional nodes. But this pushes the relevant information further into the subgraph and might therefore not be captured as good as would be without this adjustment.

## example - Barbell graph
Below a toy example is shown where GraphCASE is used to map a simple graph into a 2-dimensionan representation including information from the local neighbourhood, nodes labels and edge labels. The graph on the right show a directed graph consisting of two cliques (fully connected) of size 10 connected via a path of size 10. the nodes have two labels (refer to [Graphbell example](https://github.com/tonyPo/GraphCase/wiki/Graphbell-example) for details) and the egdes have one weight label. The edge color indicates the edge weight. The plot on the right shows the convertion of the graph into a 2 dimensional embedding. Dimension 1 is plotted on the x-axis and dimension 2 is plotted on the y-axis. The color coding of the points is equal to the color coding of the nodes in the left graph. Note that the embeding of the nodes on the path are in the upper right corner and the nodes of the cliques are in the lower left corner. This matches with our intuition as the neighbourhood of the clique nodes can be considered the opposite of the neighbourhood of the path nodes. Additionally, the representation of the clique nodes are close to each other but do slightly differ. This difference reflects the difference in node labels. 


<table style="width:100%">
  <tr>
    <th><img src="https://github.com/tonyPo/GraphCase/blob/feature/experiments/graphbell.png?raw=true" alt="Graph bell" width="350"/></th>
    <th><img src="https://github.com/tonyPo/GraphCase/blob/feature/experiments/embed_graphbell.png?raw=true" alt="Graph bell embedding" width="350"/></th>
  </tr><tr>
    <th>Graph bell</th><th>2-dim embedding of graph bell</th>
  </tr>
</table>

## example - Mirrored karate network
The below graph shows the famous Zachary's karate club network. The original network of 34 nodes is mirrored and the connected with an edge. The plot on the left shows the mirrored karate network and the right plot the corresponding embeddings calculated with GraphCASE. Not that the embedding of a node and it's corresponding mirrored node is exactly the same. The is due to the deterministic behaviour of the algorithm avoiding unnessary noise.

Note that additionally, 8 nodes (21, 9 ,20 , 17, 14, 15, 18, 12) have the same embedding. All these nodes share the some local neighbourhood, consisting of 2 direct neighbours and these direct neightbours all have 5 or more connections. The node embedding in the upper left corner are nodes with small local neighbourhood, while the node embedding in the lower right corner are nodes with a large local neighbourhood.

<table style="width:100%">
  <tr>
    <th><img src="https://github.com/tonyPo/GraphCase/blob/feature/experiments/karate_mir.png?raw=true" alt="Graph bell" width="400"/></th>
    <th><img src="https://github.com/tonyPo/GraphCase/blob/feature/experiments/karate_embed_base.png?raw=true" alt="Graph bell embedding" width="400"/></th>
  </tr><tr>
    <th>Graph bell</th><th>2-dim embedding of graph bell</th>
  </tr>
</table>


[1]:  William L Hamilton.  Inductive Representation Learning on Large Graphs.(Nips):1–11, 2017.  
[2]:  Thomas  N  Kipf  and  Max  Welling.    Semi-Supervised  Classification  withGraph Convolutional Networks.  pages 1–14, 2016.
