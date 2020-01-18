import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        print('attention layers in_features out_features', in_features, out_features)

        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear_layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.LeakyReLU())
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)


    def forward(self, input):
        #print('input shape ',input.shape)
        #input = input.squeeze()
        path_num = input.size()[0]
        # batch_W = self.W.unsqueeze(0)
        # batch_W = batch_W.repeat(path_num,1,1)
        # input = input.bmm(batch_W)
        #input = torch.mm(input, self.W) # input is 3 dimmension 10*5*8   output 10*8
        N = input.size()[1] # N = path length
        #print('input ', input.shape)
        points2 = input[0][int(N/2)].repeat(path_num, N, 1)
        #print('points2 ', points2.shape)
        #e = self.cos(input, points2)
        #e = torch.dist(input, points2, p=2)
        e = torch.pow(input - points2, 2).sum(2) 
        #print('e1 ', e.shape)
        e = e.unsqueeze(2)
        e = self.linear_layer(e)
        #e = e.unsqueeze(2)

        # a_input = torch.cat([input, input[0][int(N/2)].repeat(path_num, N, 1)], dim=2).view(path_num, N, 2 * self.out_features)
        # #print('a_input', a_input.shape)
        # e = self.leakyrelu(torch.matmul(a_input, self.a))
        # #print('e ',e.shape)

        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=1)
        #print('attention ',attention.shape)
        attention = attention.repeat(1, 1, self.out_features)
        #print('attention ',attention.shape)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = input*attention  #f you want elementwise multiplication, use the multiplication operator (*); if you want batched matrix multiplication use torch.bmm.
        #print('h_prime 1', h_prime.shape)
        h_prime = torch.sum(h_prime, 1)
        #print('h_prime ', h_prime.shape)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
