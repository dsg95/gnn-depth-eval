
from models.gnn_module import GNN
from torch_geometric.nn import GCN2Conv


class GCNII(GNN):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_layers: int,
                 alpha=0.2, dropout=0.5):

        self.alpha = alpha

        super().__init__(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                         n_layers=n_layers, conv_op=GCN2Conv, conv_args={'channels': hidden_channels,
                                                                         'alpha': alpha},
                         dropout=dropout)

    def save(self, filepath, x, y, layer):
        super().save(filepath, x, y, layer, hyperparameters={'alpha': self.alpha})
        return
