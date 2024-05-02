
from models.gnn_module import GNN

import torch
from torch import Tensor
from typing import Optional, Tuple
from torch.nn import Parameter

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_remaining_self_loops


class GCNInitResLayer(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int,
                 alpha: float,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self.edge_transform = None

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                raise NotImplementedError('GCN normalization not implemented for SparseTensor')

        x_0 = self.alpha * x_0[:x.size(0)]
        x_gcn = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        t1 = torch.mul(x_gcn, (1. - self.alpha))

        x = (x_0 + t1) @ self.weight1

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(({self.channels}), '
                f'alpha={self.alpha}')


class GCNInitRes(GNN):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, n_layers: int,
                 alpha=0.2, dropout=0.5):

        self.alpha = alpha

        super().__init__(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                         n_layers=n_layers, conv_op=GCNInitResLayer, conv_args={'channels': hidden_channels,
                                                                                'alpha': alpha},
                         dropout=dropout)

    def save(self, filepath, x, y, layer):
        super().save(filepath, x, y, layer, hyperparameters={'alpha': self.alpha})
        return
