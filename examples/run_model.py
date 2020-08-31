import os
import sys
sys.path.insert(0, os.getcwd())

import tensorflow as tf
import matplotlib.pyplot as plt
from GAE.graph_case_controller import GraphAutoEncoder
import examples.example_graph_bell_version2 as gb


G = gb.create_directed_barbell(10, 10)
gae = GraphAutoEncoder(G, support_size=[3, 3, 3], dims=[2, 6, 6, 6, 6, 1], batch_size=5,
                       max_total_steps=10, verbose=True, seed=2)

for i in range(len(gae.dims)):
    h = gae.train_layer(i+1, act=tf.nn.relu)

h = gae.train_layer(len(gae.dims), all_layers=True, act=tf.nn.relu)
# print(h1['val_l'])

e = gae.calculate_embeddings()
print(f"e: \n {e}")

# fig, ax = plt.subplots()
# ax.scatter(e[:,1], e[:,2])
# for i, txt in enumerate(e[:,0]):
#     ax.annotate(txt, (e[i,1], e[i,2]))
# plt.xlabel("Leprechauns")
# plt.ylabel("Gold")
# plt.legend(loc='upper left')
# plt.show()

# fig, ax = plt.subplots()
# ax.scatter(h['i'], h['val_l'])
# plt.xlabel("iteration")
# plt.ylabel("validaiton loss")
# plt.show()
