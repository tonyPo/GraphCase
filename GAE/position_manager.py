import numpy as np
import math
import time
import networkx as nx
import pygsp as gsp
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import partial

class PositionManager:
    """class that calculate the positional encoding that is used in the input_layer Z
    """   
    def __init__(self, G, in_sample, out_sample, hubs):
        self.in_sample = in_sample
        self.out_sample = out_sample
        self.hubs = hubs
        self.number_of_nodes = out_sample.shape[0]
        
        # rescaling factor used when creating single pos dic
        self.factor = 10**(int(math.log10(self.number_of_nodes))+1)
  
        start = time.time()
        print(f"start creating position dict at {start}")
        self.pos_dicts = self.create_pos_dicts()
        self.single_pos_dict = self.create_single_pos_dict(self.pos_dicts)
        end = time.time()
        print(f"pos dictionary created in {end-start} second: {end}")

    @staticmethod
    def select_first_embeding(emb1, emb2):
        """Compares two positions embeddings and select the one with the shortest path to the root node.

        Args:
            emb1 (array<float>): position embeddding 1
            emb2 (array<float>): position embeddding 1

        Returns:
            array<float> : postion embeding that has the closest path to the root node.
        """
        
        # extract index of the last non zero value in the list
        max_ind1 = np.where(emb1)[0].max()
        max_ind2 = np.where(emb2)[0].max()
        
        # determine the value of the last not zero dimension in the list
        max_ind_value1 = emb1[max_ind1]
        max_ind_value2 = emb2[max_ind2]
        
        # closest to root node is the one with the lowest index of the last non zero value.
        # in case of a tie, then take the higest value of the last none zero value
        if [max_ind1, 1 - max_ind_value1] < [max_ind2, 1 - max_ind_value2]:
            return emb1
        else:
            return emb2
    
    @staticmethod
    def add_layers_to_pos_dic(id, pos_dic, prefix, current_hub, in_sample, out_sample, hubs):
        """Adds the position embedding of the nodes in the local neighborhood of the root node.
        This function is call recirsively in a depth first manner and initiated with the id of the root node
        
        Args:
            id (integer): id of the node for which the neighbors are added.
            in_sample (_type_): sampled incoming neighbors per node id
            out_sample (_type_): sampled outgoing neighbors per node id
            pos_dic (_type_): dictionary with the positional embedding for all nodes in the local neighborhood of the root node 
            prefix (_type_): first part of the position embedding that is fixed
            hubs (_type_): total number of hubs in the local neighborhood
            current_hub (_type_): current hub.
        """    
        # process the incoming sampled nodes
        default = [1] * (hubs * 2 + 2)  # default value for position embedding when not yet in dict.
        sample = in_sample[id]
        in_lenght = (hubs - current_hub) * 2 + 1
        for i, nn in enumerate(sample):
            # update position embedding with the lowest value as this value has the shortest path to the root node
            # position encoding is prefix + 1 + 0 for remaining part of the vector
            this_position = prefix + [1 - i / sample.shape[0]] + [0] * in_lenght
            pos_emb =  PositionManager.select_first_embeding(pos_dic.get(nn, default), this_position)
            pos_dic[nn] = pos_emb
            if current_hub < hubs:  # sample the next hub
                PositionManager.add_layers_to_pos_dic(
                    nn, pos_dic, pos_emb[: 2 * current_hub + 1], current_hub + 1, in_sample, out_sample, hubs
                )
            
        # process the outcoming sampled nodes
        sample = out_sample[id]
        out_lenght = (hubs - current_hub) * 2
        for i, nn in enumerate(sample):
            # update position embedding with the lowest value as this value has the shortest path to the root node
            # position encoding is prefix + 1 + 0 for remaining part of the vector
            this_position = prefix + [0] + [1 - i / sample.shape[0]] + [0] * out_lenght
            pos_emb =  PositionManager.select_first_embeding(pos_dic.get(nn, default), this_position)
            pos_dic[nn] = pos_emb
            if current_hub < hubs:  # sample the next hub
                PositionManager.add_layers_to_pos_dic(
                    nn, pos_dic, pos_emb[: 2 * current_hub + 1], current_hub + 1, in_sample, out_sample, hubs
                )
            
    
    def create_pos_dicts(self):
        """creates a dictionary of dictionary with the embedding encoding of the nodes in the local neighborhood per root
        The outer dict keys are the root nodes, the inner dicts keys are the nodes from the local neighborhood for that root node
        and the value of the inner dicts is the positional encoding of that node in related to the root node.

        Returns:
            _type_: dict with dict containing root node, local node and value the positional enconding.
        """
        pos_dicts = {}  # dictionary to hold the dictonaries for all nodes
        
        pool_function = partial(
            PositionManager.create_pos_dicts_stub, 
            in_sample=self.in_sample, out_sample=self.out_sample, hubs=self.hubs, number_of_nodes=self.number_of_nodes
            )
        # for id in range(self.number_of_nodes):  # for all node ids + dummy node id
        for res in Parallel(n_jobs=-1)(delayed(pool_function)(i) for i in range(self.number_of_nodes)):
                # add dict to pos_dicts
                pos_dicts[res[0]] = res[1]
        
        return pos_dicts
    
    @staticmethod
    def create_pos_dicts_stub(id, in_sample, out_sample, hubs, number_of_nodes):
         # instantiate dictionary for position embedding for this root node.
        pos_dic = {}
        current_hub = 1
        prefix = [1.0]  # only the first dimension of the embedding is set to 1
        PositionManager.add_layers_to_pos_dic(id, pos_dic, prefix, current_hub, in_sample, out_sample, hubs)
    
        # overwrite the position embedding for the root 
        pos_dic[id] = [1.0] + [0] * (hubs * 2)
        pos_dic[number_of_nodes - 1] = [0] * (hubs * 2 + 1)
        return (id, pos_dic)
        
    def create_single_pos_dict(self, pos_dicts):
        """maps the two nested dicts to on dict.
        merging scheme is key1 * factor + key2

        Args:
            pos_dicts (_type_): dict of dicts

        Returns:
            _type_: dict with key node to node and value position encoding
        """
        
        new_dict = {}

        for k,dic2 in pos_dicts.items():
            for k2, v in dic2.items():
                new_dict[k * self.factor + k2] = v
                
        return new_dict
        
        
class WaveLetPositionManager:
    """class that calculates the positional encoding based on wavelets
    The positional encoding is used in the input_layer Z
    """   
    def __init__(self, G, in_sample=None, out_sample=None, hubs=2):
        self.G = G
        self.hubs = hubs
        self.dummy_id = G.number_of_nodes()
        self.Nf = 3
        self.scale = 0.5

        # rescaling factor is used to multiple the source node id before combining with the target node id to create one unique id
        self.factor = 10**(int(math.log10(G.number_of_nodes()))+1)

        start = time.time()
        print(f"start creating position dict at {start}")
        self.pos_dicts, min_val, max_val = self.create_pos_dicts()
        self.single_pos_dict = self.create_single_pos_dict(self.pos_dicts, min_val, max_val)
        end = time.time()
        print(f"pos dictionary created in {end-start} second: {end}")   

    def create_pos_dicts(self):
        """creates a dictionary of dictionary with the embedding encoding of the nodes in the local neighborhood per root
        The outer dict keys are the root nodes, the inner dicts keys are the nodes from the local neighborhood for that root node
        and the value of the inner dicts is the positional encoding of that node in related to the root node.
        
        Dummy node values are set to zero

        Returns:
            _type_: dict with dict containing root node, local node and value the positional enconding.
        """
        pos_dicts = {}  # dictionary to hold the dictonaries for all nodes
        min_val = float('inf')
        max_val = float('-inf')
        
        for id in list(self.G.nodes()):  # for all node ids excl dummy node id
            # get in and out egonets for node = id
            g_in, g_out = self._get_egonets(id)

            # create calculate wavelet coefs
            wave_coef_in, wave_coef_out = self._get_wave_coef(g_in, g_out, id)
            max_val = max(max_val, wave_coef_in.max(), wave_coef_out.max())
            min_val = min(min_val, wave_coef_in.min(), wave_coef_out.min())

            #create dict {node: embedding}
            pos_dic = {}
            for i,n in enumerate(g_in.nodes()):
                pos_embed = np.concatenate([wave_coef_in[i], wave_coef_out[i]],axis=0)
                pos_dic[n] = list(pos_embed)

            #add dummy node
            pos_dic[self.dummy_id] = [0] * (2 * self.Nf)

            # add dict to pos_dicts
            pos_dicts[id] = pos_dic

        # add dummy node, has only dummy to dummy edges
        pos_dic = {self.dummy_id: [0] * (2 * self.Nf)}
        pos_dicts[self.dummy_id] = pos_dic
                  
        return pos_dicts, min_val, max_val

    def _get_wave_coef(self, g_in, g_out, root_node):
        """determindes the wavelt coefficient for the in and out going networks 
        localized at the rootnode

        Args:
            g_in (nx.Graph): undirect graph with weight adjusted for incoming edges
            g_out (nx.Graph): undirect graph with weight adjusted for outgoing edges
            root_node_id (integer): root node

        Returns:
            tupple of 2 arrays: array with incoming and outgoing wavelet coefficients.
        """
        # filter out single nodes.
        if g_in.number_of_nodes() == 1:
            return (np.array([[0] * self.Nf]), np.array([[0] * self.Nf]))
        
        #create pygsp graph
        adj_in = nx.adjacency_matrix(g_in)
        adj_out = nx.adjacency_matrix(g_out)
        g_in_gsp = gsp.graphs.Graph(adj_in)
        g_out_gsp = gsp.graphs.Graph(adj_out)
        g_in_gsp.compute_fourier_basis()
        g_out_gsp.compute_fourier_basis()
        
        #setup filters
        f_wave_in = gsp.filters.Meyer(g_in_gsp, Nf=self.Nf)
        f_wave_out = gsp.filters.Meyer(g_out_gsp, Nf=self.Nf)
        
        #apply filers
        root_node_id = list(g_in.nodes()).index(root_node)
        wave_coef_in = f_wave_in.localize(root_node_id)
        wave_coef_out = f_wave_out.localize(root_node_id)
        
        return (wave_coef_in, wave_coef_out)

    def _get_egonets(self, id):
        """Creates a egonet for outgoing and incoming edges. To ensure that all nodes are presents
        in both in and out ego net. We use the directed net to reweight the edges of the undirected
        net. The weightst are adjusted by w= W_undirected * exp ^ w_directed

        Args:
            id (integer): root node
        """
        # G_ego = nx.ego_graph(self.G, id, radius=self.hubs, undirected=True)
        # # need to average weight in case of both in and outgoing node to same neighbor
        # for s,d,w in G_ego.edges(data='weight'):
        #     if G_ego.has_edge(d,s):
        #         new_w = (w + G_ego[d][s]['weight'])
        #         G_ego[d][s]['weight'] = new_w
        #         G_ego[s][d]['weight'] = new_w
        # G_ego = G_ego.to_undirected()
        G_und = nx.ego_graph(self.G.to_undirected(), id, radius=self.hubs)

        #create out egonet
        '''For cycles, the direct ego graph is missing the return path edges'''
        G_ego_out = G_und.copy()
        G_ego_out_factor = nx.ego_graph(self.G, id, radius=self.hubs)
        for s,d in G_und.edges(data=False):
            if G_ego_out_factor.has_edge(s,d):
                G_ego_out[s][d]['weight'] = math.exp(G_ego_out_factor[s][d]['weight']) / math.e
            else:
                G_ego_out[s][d]['weight'] = 1/ math.e

        #create in egonet
        G_ego_in = G_und.copy()
        G_ego_in_factor = nx.ego_graph(self.G.reverse(), id, radius=self.hubs)
        for s,d in G_und.edges(data=False):
            if G_ego_in_factor.has_edge(s,d):
                G_ego_in[s][d]['weight'] = math.exp(G_ego_in_factor[s][d]['weight']) / math.e
            else:
                G_ego_in[s][d]['weight'] = 1/ math.e
            
        return (G_ego_in, G_ego_out) 
 
    def create_single_pos_dict(self, pos_dicts, min_val, max_val):
        """maps the two nested dicts to on dict.
        merging scheme is key1 * factor + key2

        Args:
            pos_dicts (_type_): dict of dicts

        Returns:
            _type_: dict with key node to node and value position encoding
        """

        new_dict = {}

        for k,dic2 in pos_dicts.items():
            for k2, emb in dic2.items():
                new_dict[k * self.factor + k2] = [(v-min_val)/(max_val - min_val) * self.scale for v in emb]
                
        return new_dict
