import torch.nn as nn
from torch_geometric.nn import MLP, GAT
from .gine import GINE
from .graphtransformer import GraphTransformerLayer


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
        phi_out = self.phi(x, data.edge_index, data.edge_attr) + self.phi(-x, data.edge_index, data.edge_attr)
        out = self.rho(phi_out)
        return out
    

class SignNetLayer_Transformer(nn.Module):
    def __init__(self, pos_enc_dim, hidden_dim, out_dim, edge_dim):
        super().__init__()
        self.phi = GINE(in_channels=pos_enc_dim, hidden_channels=hidden_dim, out_channels=out_dim, 
                        edge_dim=edge_dim, num_layers=3, jk='lstm')
        
        self.rho = GraphTransformerLayer(in_dim=out_dim, out_dim=out_dim, num_heads=8, 
                                         batch_norm=True, residual=True, use_bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def forward(self, data):
        x = data.pos_enc
        phi_out = self.phi(x, data.edge_index, data.edge_attr) + self.phi(-x, data.edge_index, data.edge_attr)
        out = self.rho(phi_out, data.edge_index)
        return out