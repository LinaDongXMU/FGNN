import torch
import torch.nn as nn
from torch_geometric.nn import MLP, MessagePassing
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models.basic_gnn import BasicGNN


class SignNetLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_dim):
        super().__init__()
        self.phi = GINE(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=out_dim, 
                        edge_dim=edge_dim, num_layers=3, jk='lstm')
        
        self.rho = MLP([out_dim, hidden_dim, out_dim])
        self.reset_parameters()

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def forward(self, data):
        x = data.pos_enc
        phi_out = self.phi(x, data.edge_index, data.edge_attr.float()) + self.phi(-x, data.edge_index, data.edge_attr.float())
        out = self.rho(phi_out)
        return out  

class GINEConv(MessagePassing):
    def __init__(self, nn, eps=0., train_eps=False,
                 edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            in_channels = self.nn.in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None):
        """"""
        if isinstance(x, torch.Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j, edge_attr):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
    
class GINE(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, edge_dim: int, **kwargs) -> MessagePassing:
        mlp = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        return GINEConv(mlp, train_eps=True, edge_dim=edge_dim, **kwargs)