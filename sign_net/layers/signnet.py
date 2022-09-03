import torch
import torch.nn as nn
import torch.functional as F
from torch_geometric.nn import MLP
from torch_scatter import scatter
from .gine import GINE

from utils import to_dense_list_EVD


class SignNetLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim, edge_dim, num_layer=3, eigenval_dim=1):
        super().__init__()
        self.phi = GINE(eigenval_dim, hidden_dim, num_layer, out_channels=edge_dim, edge_dim=edge_dim) 
        self.rho = MLP([edge_dim, hidden_dim, out_dim])
        self.reset_parameters()

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        x = eigV_dense.unsqueeze(-1).transpose(0,1)
        phi_out = self.phi(x, data.edge_index, data.edge_attr.float()) + self.phi(-x, data.edge_index, data.edge_attr.float())
        phi_out = phi_out.transpose(0,1)
        phi_out = torch.sum(phi_out, dim=1).squeeze(1)
        out = self.rho(phi_out)
        return out