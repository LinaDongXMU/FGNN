import torch.nn as nn
from torch_geometric.nn import MLP
from .gine import GINE


class SignNetLayer(nn.Module):
    def __init__(self, pos_enc_dim, hidden_dim, out_dim, edge_dim):
        super().__init__()
        self.phi = GINE(in_channels=pos_enc_dim, hidden_channels=hidden_dim, out_channels=out_dim, 
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