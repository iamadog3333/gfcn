"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


import my_data_loader_just_paths

path_loader = my_data_loader_just_paths.my_data_set(dataset_name='citeseer')



def get_cnn_input_of_a_node(node_num, x):
    paths = path_loader.get_paths_of_a_node(node_num)
    #print('paths ', paths)
    cnn_input = []
    for path in paths:
        one_path_data = []
        for node in path:
            one_path_data.append(x[node])
        one_path_data = torch.stack(one_path_data)
        cnn_input.append(one_path_data)
    #print('cnn_input ', cnn_input)
    cnn_input = torch.stack(cnn_input)
    #print('cnn_input ', cnn_input)
    #print('cnn_input ', cnn_input.shape) #cnn_input shape torch.Size([10, 5, 7])
    return cnn_input

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 cnn_channle):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

        self.cnn_layer = nn.Sequential(
            nn.Conv2d(10, cnn_channle, kernel_size=(3,1), stride=(1,1), padding=(0,0)),    #in_channels, out_channels, kernel_size
            #nn.BatchNorm1d(feature_final_len*2),
            nn.LeakyReLU(), #nn.ReLU(),
            nn.Conv2d(cnn_channle, 1, kernel_size=(3,1), stride=(1,1), padding=(0,0)),    #in_channels, out_channels, kernel_size
            #nn.BatchNorm1d(feature_final_len*2),
            nn.LeakyReLU(), #nn.ReLU(),
            )

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)


        #gfcn part
        x = logits
        num_of_node = x.shape[0]
        #print('x shape ', x.shape)
        #print('num_of_node', num_of_node)
        node_vecs = []
        for node_num in range(num_of_node):
            cnn_input = get_cnn_input_of_a_node(node_num, x)
            cnn_input = torch.unsqueeze(cnn_input, 0) # add a batch axis
            #print('cnn_input shape', cnn_input.shape)
            cnn_out = self.cnn_layer(cnn_input)
            #print('cnn_output shape', cnn_out.shape)
            cnn_out = torch.squeeze(cnn_out)
            node_vecs.append(cnn_out)
        x = torch.stack(node_vecs) 
        logits = x   

        return logits
