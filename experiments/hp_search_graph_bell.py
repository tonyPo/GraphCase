#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18-09-2020
@author: tonpoppe
"""

import os
import pickle
import sys
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.getcwd()+"/examples")
print(os.getcwd())
from  GAE.graph_case_controller import GraphAutoEncoder
import example_graph_bell_version2 as gb

space = {'dim0': hp.quniform('dim0', 1, 3, 1),
         'dim1': hp.quniform('dim1', 2, 8, 1),
         'dim2': hp.quniform('dim2', 2, 8, 1),
         'dim3': hp.quniform('dim3', 2, 6, 1)}
        #  'learning_rate' : hp.loguniform('learning_rate', -5, -4)}

def train_model(params):
    """
    function to create and train the model. This is called by hyper opt.
    It returned the loss (=optimisation metric), status and a dict with
    supporting information.
    """
    dims = [int(params['dim0']), int(params['dim1']), int(params['dim2']), int(params['dim3']), 2]

    gae = GraphAutoEncoder(G, learning_rate=0.01, support_size=[5, 5],
                           dims=dims, batch_size=12, max_total_steps=250)

    train_res = {}
    for i in range(len(gae.dims)):
        train_res["l"+str(i+1)] = gae.train_layer(i+1, act=tf.nn.relu)

    train_res['all'] = gae.train_layer(len(gae.dims), all_layers=True, act=tf.nn.relu)                       

    loss_val = train_res['all']['val_l'][-3:]
    print(f"loss val {loss_val}")
    loss = sum(loss_val) / len(loss_val)
    train_res['loss'] = loss

    return {'loss': loss, 'status': STATUS_OK, 'hist':train_res}

def execute_grid_search(file_name):
    # function to call the grid search
    trials = Trials()

    argmin = fmin(fn=train_model,
                  space=space,
                  algo=tpe.suggest,
                  max_evals=50,
                  trials=trials,
                  rstate=np.random.RandomState(1))

    pickle.dump(trials, open(file_name, "wb"))
    return trials

G = gb.create_directed_barbell(10, 10)
res = execute_grid_search("/Users/tonpoppe/grid_search_1")
print("trials are finished")

print(res.best_trial)
#%%
import pickle
train_res = pickle.load(open("/Users/tonpoppe/grid_search_1", "rb"))
print(train_res.best_trial['misc'])
# %%
