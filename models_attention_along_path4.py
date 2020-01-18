import torch
import torch.nn as nn
import torch.nn.functional as F
#from layers import GraphAttentionLayer, SpGraphAttentionLayer
from layers_att_along_path4 import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #self.out_att = GraphAttentionLayer(nhid* nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, is_training=True):
        x = x.squeeze(0)
        #x = F.dropout(x, self.dropout, training=is_training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        #x = F.dropout(x, self.dropout, training=is_training)

        #x = x.unsqueeze(0)
        #x = F.elu(self.out_att(x))

        #add vectors from different paths
        x = F.elu(x)
        #print('x 1', x.shape)
        x = torch.sum(x, 0, keepdim=True)  # x shape 10*8---> 1*8
        #print('x ', x.shape)
        #x = x.unsqueeze(0)

        return F.log_softmax(x, dim=1) #F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, is_training):
        x = F.dropout(x, self.dropout, training=is_training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=is_training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

