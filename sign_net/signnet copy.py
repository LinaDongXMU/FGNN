import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GIN, MLP

from utils import *

class SignNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer=3):
        super().__init__()
        self.phi = GIN(in_dim, hidden_dim, num_layer, out_dim) 
        self.rho = MLP([out_dim, hidden_dim, in_dim])
        self.reset_parameters()

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        x = eigV_dense
        x = self.phi(x, data.edge_index, data.edge_attr) + self.phi(-x, data.edge_index, data.edge_attr)
        x = self.rho(x)
        return x

class SignNetGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_signnet, nl_gnn, nl_rho=4, ignore_eigval=False, gnn_type='GINEConv'):
        super().__init__()
        self.sign_net = SignNet(n_hid, nl_signnet, nl_rho=4, ignore_eigval=ignore_eigval)
        self.gnn = GIN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type) 

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        pos = self.sign_net(data)
        return self.gnn(data, pos)