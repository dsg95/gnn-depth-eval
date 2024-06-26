#
# =====================================================================================================
# GRADIENT GATING Implementation by Konstantin Rusch (2022)
#
# Extracted from https://github.com/tk-rusch/gradientgating
# MIT License
#
# Citation
# Rusch, T. K.; Chamberlain, B. P.; Mahoney, M. W.; Bronstein, M. M.; and Mishra, S. 2023. Gradient Gating for Deep
# Multi-Rate Learning on Graphs. In The Eleventh International Conference on Learning Representations.
#
# =====================================================================================================
#


import torch
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

import os
from os.path import isfile
from models.gnn_module import GNN


class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='GCN', activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0,dim_size=X.size(0), reduce='mean'))

        return gg


class G2_GNN(GNN):
    def __init__(self, nfeat, nhid, nclass, nlayers, conv_type='GCN', p=2., drop_in=0.5, drop=0.5, use_gg_conv=True):
        super(G2_GNN, self).__init__(in_channels=nfeat, out_channels=nclass, hidden_channels=nhid, n_layers=nlayers,
                                     conv_op=None, conv_args={})
        self.conv_type = conv_type
        self.enc = nn.Linear(nfeat, nhid)
        self.dec = nn.Linear(nhid, nclass)
        self.drop_in = drop_in
        self.drop = drop
        self.nlayers = nlayers
        if conv_type == 'GCN':
            self.conv = GCNConv(nhid, nhid)
            if use_gg_conv == True:
                self.conv_gg = GCNConv(nhid, nhid)
        elif conv_type == 'GAT':
            self.conv = GATConv(nhid,nhid,heads=4,concat=True)
            if use_gg_conv == True:
                self.conv_gg = GATConv(nhid,nhid,heads=4,concat=True)
        else:
            print('specified graph conv not implemented')

        if use_gg_conv == True:
            self.G2 = G2(self.conv_gg,p,conv_type,activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv,p,conv_type,activation=nn.ReLU())

    def forward(self, data, save_as=''):

        if save_as != '' and isfile(save_as):
            os.remove(save_as)

        X = data.x
        n_nodes = X.size(0)
        edge_index = data.edge_index
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        if save_as != '':
            self.save(save_as, X[data.test_mask], data.y[data.test_mask], 0)

        for i in range(self.nlayers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
            tau = self.G2(X, edge_index)
            X = (1 - tau) * X + tau * X_

            if save_as != '':
                self.save(save_as, X[data.test_mask], data.y[data.test_mask], i+1)
        X = F.dropout(X, self.drop, training=self.training)

        return self.dec(X)

