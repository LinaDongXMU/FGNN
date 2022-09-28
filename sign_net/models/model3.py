import torch.nn as nn
from torch_geometric.nn import global_mean_pool, AttentiveFP
from layers import EGNNLayer, RegressionLayer, SignNetLayer, SignNetLayer_Transformer
from utils import *


class DrugNet(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.sign_net = SignNetLayer_Transformer(pos_enc_dim, hidden_dim, out_dim, edge_dim)
        
        self.gnn = AttentiveFP(in_channels=node_dim+out_dim, hidden_channels=hidden_dim, 
                               out_channels=out_dim, edge_dim=rbf_dim, num_layers=3, num_timesteps=3)
        
        self.regression = RegressionLayer(out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        dist_rbf = data.dist_rbf.float()
        pos = self.sign_net(data)   # node_num, 10
        x = torch.cat([node_feat, pos], dim=-1)
        graph = self.gnn(x, edge_index, dist_rbf, data.batch)
        out = self.regression(graph)
        return out