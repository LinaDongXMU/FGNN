import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GINEConv
from torch_geometric.nn.inits import reset


class RegressionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        out = self.fc_net(x)
        return out


class MaskedBN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        
    def reset_parameters(self):
        self.bn.reset_parameters() 
        
    def forward(self, x, mask=None):
        if mask is None:
            return self.bn(x.transpose(1,2)).transpose(1,2)
        x[mask] = self.bn(x[mask])
        return x


class MaskedLN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=1e-6)
        
    def reset_parameters(self):
        self.ln.reset_parameters() 
        
    def forward(self, x, mask=None):
        if mask is None:
            return self.ln(x)
        x[mask] = self.ln(x[mask])
        return x


class MaskedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, nlayer=2, with_final_activation=True, with_norm=True, bias=True, hid_dim=None):
        super().__init__()
        
        hid_dim = in_dim if hid_dim is None else hid_dim
        self.layers = nn.ModuleList([nn.Linear(in_dim if i==0 else hid_dim, 
                                               hid_dim if i<nlayer-1 else out_dim, 
                                               bias=True if (bias) or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([MaskedBN(hid_dim if i<nlayer-1 else out_dim) if with_norm else nn.Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x, mask=None):
        # x: n x k x d
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            
            if mask is not None:
                x[~mask] = 0
                
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x, mask)
                x = F.relu(x)  
        return x    


class MaskedGINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhid=None):
        super().__init__()
        self.nn = MaskedMLP(nin, nout, 2, False, bias=bias, nhid=nhid)
        self.layer = GINEConv(nn.Identity(), train_eps=True)
        
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
        
    def forward(self, x, edge_index, edge_attr, mask=None):
        assert x[~mask].max() == 0
        x = self.layer(x, edge_index, edge_attr)
        
        if mask is not None:
            x[~mask] = 0 

        x = self.nn(x, mask)
        return x
    
    
# class MaskedGINEConv(MessagePassing):
#     def __init__(self, in_dim, out_dim, eps=0., train_eps=True, edge_dim=None, **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self.nn = MaskedMLP(in_dim, out_dim, nlayer=2, with_final_activation=False)
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.Tensor([eps]))
#         else:
#             self.register_buffer('eps', torch.Tensor([eps]))
#         if edge_dim is not None:
#             if hasattr(self.nn[0], 'in_features'):
#                 in_channels = self.nn[0].in_features
#             else:
#                 in_channels = self.nn[0].in_channels
#             self.lin = nn.Linear(edge_dim, in_channels)
#         else:
#             self.lin = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.nn)
#         self.eps.data.fill_(self.initial_eps)
#         if self.lin is not None:
#             self.lin.reset_parameters()

#     def forward(self, x, edge_index, edge_attr=None, mask=None, size=None):
#         assert x[~mask].max() == 0
#         x = (x, x)
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

#         x_r = x[1]
#         if x_r is not None:
#             out += (1 + self.eps) * x_r

#         return self.nn(out)

#     def message(self, x_j, edge_attr):
#         if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
#             raise ValueError("Node and edge feature dimensionalities do not "
#                              "match. Consider setting the 'edge_dim' "
#                              "attribute of 'GINEConv'")

#         if self.lin is not None:
#             edge_attr = self.lin(edge_attr)

#         return (x_j + edge_attr).relu()