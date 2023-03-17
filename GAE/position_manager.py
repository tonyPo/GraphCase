import numpy as np
import math
import time

class PositionManager:
    """class that calculate the positional encoding that is used in the input_layer Z
    """   
    def __init__(self, G, in_sample, out_sample, hubs):
        self.G = G
        self.in_sample = in_sample
        self.out_sample = out_sample
        self.hubs = hubs
        
        # rescaling factor used when creating single pos dic
        self.factor = 10**(int(math.log10(G.number_of_nodes()))+1)
        
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
         
    def add_layers_to_pos_dic(self, id, pos_dic, prefix, current_hub):
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
        default = [1] * (self.hubs * 2 + 1)  # default value for position embedding when not yet in dict.
        sample = self.in_sample[id]
        in_lenght = (self.hubs - current_hub) * 2 + 1
        for i, nn in enumerate(sample):
            # update position embedding with the lowest value as this value has the shortest path to the root node
            # position encoding is prefix + 1 + 0 for remaining part of the vector
            this_position = prefix + [1 - i / sample.shape[0]] + [0] * in_lenght
            pos_emb =  self.select_first_embeding(pos_dic.get(nn, default), this_position)
            pos_dic[nn] = pos_emb
            if current_hub < self.hubs:  # sample the next hub
                self.add_layers_to_pos_dic(nn, pos_dic, pos_emb[: 2 * current_hub + 1], current_hub + 1)
            
        # process the outcoming sampled nodes
        sample = self.out_sample[id]
        out_lenght = (self.hubs - current_hub) * 2
        for i, nn in enumerate(sample):
            # update position embedding with the lowest value as this value has the shortest path to the root node
            # position encoding is prefix + 1 + 0 for remaining part of the vector
            this_position = prefix + [0] + [1 - i / sample.shape[0]] + [0] * out_lenght
            pos_emb =  self.select_first_embeding(pos_dic.get(nn, default), this_position)
            pos_dic[nn] = pos_emb
            if current_hub < self.hubs:  # sample the next hub
                self.add_layers_to_pos_dic(nn, pos_dic, pos_emb[: 2 * current_hub + 1], current_hub + 1)
            
    
    def create_pos_dicts(self):
        """creates a dictionary of dictionary with the embedding encoding of the nodes in the local neighborhood per root
        The outer dict keys are the root nodes, the inner dicts keys are the nodes from the local neighborhood for that root node
        and the value of the inner dicts is the positional encoding of that node in related to the root node.

        Returns:
            _type_: dict with dict containing root node, local node and value the positional enconding.
        """
        pos_dicts = {}  # dictionary to hold the dictonaries for all nodes
        
        for id in list(self.G.nodes()) + [self.G.number_of_nodes()]:  # for all node ids + dummy node id
            # instantiate dictionary for position embedding for this root node.
            pos_dic = {}
            current_hub = 1
            prefix = [1.0]  # only the first dimension of the embedding is set to 1
            self.add_layers_to_pos_dic(id, pos_dic, prefix, current_hub)
        
            # overwrite the position embedding for the root 
            pos_dic[id] = [1.0] + [0] * (self.hubs * 2)
            pos_dic[self.G.number_of_nodes() - 1] = [0] * (self.hubs * 2 + 1)
            
            # add dict to pos_dicts
            pos_dicts[id] = pos_dic
            
        return pos_dicts
        
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
        
        