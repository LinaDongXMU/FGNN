import torch.nn as nn
from torch_geometric.nn import MLP, GAT
from .gine import GINE
from .graphtransformer import GraphTransformerLayer, MultiHeadAttentionLayer
from .graphtransformer_edge import MultiHeadAttentionLayer_edge


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
        self.out_dim = out_dim
        self.phi = GINE(in_channels=pos_enc_dim, hidden_channels=hidden_dim, out_channels=out_dim, 
                        edge_dim=edge_dim, num_layers=3, jk='lstm')
        
        # self.rho = GraphTransformerLayer(in_dim=out_dim, out_dim=out_dim, num_heads=8, 
        #                                  batch_norm=True, residual=True, use_bias=True)
        self.rho = MultiHeadAttentionLayer(in_dim=out_dim, out_dim=out_dim//8, num_heads=8)
        self.reset_parameters()

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def forward(self, data):
        x = data.pos_enc
        phi_out = self.phi(x, data.edge_index, data.edge_attr) + self.phi(-x, data.edge_index, data.edge_attr)
        out = self.rho(phi_out, data.edge_index).view(-1, self.out_dim)
        return out
    
    
class SignNetLayer_Transformer2(nn.Module):
    def __init__(self, pos_enc_dim, hidden_dim, out_dim, edge_dim):
        super().__init__()
        self.out_dim = out_dim
        self.phi = MultiHeadAttentionLayer_edge(in_dim=pos_enc_dim, out_dim=out_dim//8, num_heads=8, edge_dim=edge_dim, using_bias=True)
        
        self.rho = MLP([out_dim, hidden_dim, out_dim])
        self.reset_parameters()

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def forward(self, data):
        x = data.pos_enc
        phi_out = self.phi(x, data.edge_attr, data.edge_index) + self.phi(-x, data.edge_attr, data.edge_index)
        phi_out = phi_out[0].view(-1, self.out_dim)
        out = self.rho(phi_out)
        return out