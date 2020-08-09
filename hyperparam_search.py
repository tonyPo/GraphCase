from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
from GAE.graphCaseController import GraphAutoEncoder
import examples.example_graph_bell_version2 as gb
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle


space = {
  'dim0': hp.quniform('dim0', 1, 3, 1),
  'dim1': hp.quniform('dim1', 2, 10, 1),
  'dim2': hp.quniform('dim2', 2, 10, 1),
  'learning_rate' : hp.loguniform('learning_rate', -5, -3)
}

def train_model(params):
    """
    function to create and train the model. This is called by hyper opt.
    It returned the loss (=optimisation metric), status and a dict with 
    supporting information.
    """
    dims = [int(params['dim0']), int(params['dim1']), int(params['dim2']), 1]

    gae = GraphAutoEncoder(G, learning_rate=params['learning_rate'], 
                            support_size=[5,5], dims=dims, batch_size=2, max_total_steps=10000)
    h={}
    h['l1'] = gae.train_layer(1,act=tf.nn.relu)
    h['l2'] = gae.train_layer(2,act=tf.nn.relu)
    h['l3'] = gae.train_layer(3,act=tf.nn.relu)
    h['l4'] = gae.train_layer(4,act=tf.nn.relu)
    h['all'] = gae.train_layer('all',act=tf.nn.relu)

    loss_val = h['all']['val_l'][-3:]
    print(f"loss val {loss_val}")
    loss = sum(loss_val) / len(loss_val)
    h['loss'] = loss

    return {'loss': loss, 'status': STATUS_OK, 'hist':h}

def execute_grid_search(file_name):
    # function to call the grid search
    trials = Trials()

    argmin = fmin(
    fn=train_model,
    space=space,
    algo=tpe.suggest,
    max_evals=5,
    trials=trials,
    rstate = np.random.RandomState(1)
    )
    
    pickle.dump(trials, open(file_name, "wb"))
    return trials

    
G = gb.create_directed_barbell(10, 10)
trials = execute_grid_search("grid_search_1")
print(f"trials are: {trials}")

