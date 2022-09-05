import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, MessagePassing, global_mean_pool
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import GINEConv
from torch_geometric.nn.models import AttentiveFP, GCN
from torch_geometric.nn import BatchNorm, MessagePassing, global_mean_pool
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset

class Regression_Linear(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(in_dim,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,1),
        )
    def forward(self,x):
        out = self.fc_net(x)
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
    def init_conv(self, in_channels: int, out_channels: int, edge_dim: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP([in_channels, out_channels, out_channels], batch_norm=True)
        return GINEConv(mlp, train_eps=True, edge_dim=edge_dim, **kwargs)


class DrugNet(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim, edge_dim, num_layers, drop_ratio):
        super(DrugNet, self).__init__()
        #self.convs1 = AttentiveFP(emb_dim, hidden_dim, out_dim, 1, num_layers, num_timesteps=1, dropout=drop_ratio)
        
        self.convs1 = AttentiveFP(emb_dim, hidden_dim, out_dim, edge_dim=edge_dim, 
                                 num_layers=num_layers, num_timesteps=3)
        self.convs2 = GINE(emb_dim, hidden_dim, num_layers, out_dim, edge_dim=edge_dim, jk='lstm')                         
        #self.convs2 = GIN(emb_dim, out_dim, num_layers, out_dim, jk='lstm', dropout=drop_ratio)
        self.fc1 = MLP([128,256,128], batch_norm=True)
        self.fc2 = MLP([128,256,128], batch_norm=True)
        self.fc3 = Regression_Linear(512)
        
    def forward(self, x, edge_index, edge_attr, batch):
        #batch=batch.to(torch.long)
        edge_attr = F.normalize(edge_attr, p=2, dim=1)
        graph_1 = self.convs1(x, edge_index, edge_attr, batch)
        #graph_2_node = self.convs2(x, edge_index)
        #graph_2 = global_mean_pool(graph_2_node, batch)
        graph_2_node = self.convs2(x, edge_index, edge_attr)
        graph_2 = global_mean_pool(graph_2_node, batch)
        output1 = self.fc1(graph_1)
        output2 = self.fc2(graph_2)
        cat = torch.cat([graph_1, output1, graph_2, output2],dim=1)
        output = self.fc3(cat)
        
        return output.squeeze()
