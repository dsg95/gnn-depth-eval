import os
import pandas as pd
from os.path import isfile

import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_layers: int,
                 conv_op: nn.Module, conv_args: dict, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.dropout = dropout

        # Layers
        if conv_op is not None:
            if hidden_channels is not None:
                self.pre_lin = nn.Linear(in_channels, hidden_channels)
            else:
                self.hidden_channels = in_channels
                self.pre_lin = None
            self.convs = nn.ModuleList()
            for i in range(n_layers):
                conv = conv_op(**conv_args)
                self.convs.append(conv)
            self.post_lin = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, data, save_as=''):
        if save_as != '' and isfile(save_as):
            os.remove(save_as)

        _layers = []
        x, edge_index = data.x, data.edge_index
        if self.pre_lin is not None:
            x = F.dropout(x, self.dropout, training=self.training)
            x_out = self.pre_lin(x).relu()
        else:
            x_out = x
        _layers.append(x_out)
        if save_as != '':
            self.save(save_as, x_out[data.test_mask], data.y[data.test_mask], 0)
        for i, conv in enumerate(self.convs):
            x_out = F.dropout(x_out, p=self.dropout, training=self.training)
            x_out = conv(x_out, _layers[0], edge_index=edge_index)
            x_out = F.relu(x_out)

            if save_as != '':
                self.save(save_as, x_out[data.test_mask], data.y[data.test_mask], i+1)

        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        x = self.post_lin(x_out)
        return F.log_softmax(x, dim=1)

    def save(self, filepath, x, y, layer, hyperparameters={}):
        feats = ['feat_%i' % (i+1) for i in range(x.size(1))]
        cols = ['in_dim', 'out_dim', 'hidden_dim', 'n_layers', 'layer', 'y']
        cols.extend(feats)
        cols.extend(list(hyperparameters.keys()))

        df = pd.DataFrame(columns=cols)
        df[feats] = x.detach().cpu().numpy()
        df['y'] = y.detach().cpu().numpy()
        df['in_dim'] = self.in_channels
        df['out_dim'] = self.out_channels
        df['hidden_dim'] = self.hidden_channels
        df['n_layers'] = self.n_layers
        df['layer'] = layer

        for param in hyperparameters.keys():
            df[param] = hyperparameters[param]

        if isfile(filepath):
            df.to_csv(filepath, header=False, index=False, mode='a')
        else:
            df.to_csv(filepath, header=True, index=False)

        return