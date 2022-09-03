import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from layers import EGNNLayer, RegressionLayer, SignNetLayer
from utils import *


class DrugNet(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.sign_net = SignNetLayer(hidden_dim, out_dim, edge_dim, num_layer=3)
        self.gnn = EGNNLayer(node_dim+out_dim, 256, edge_dim=rbf_dim) 
        self.regression = RegressionLayer(node_dim+out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        coords = data.coords
        dist_rbf = data.dist_rbf.float()
        pos = self.sign_net(data)   # node_num, 10
        x = torch.cat([node_feat, pos], dim=-1)
        node, coords = self.gnn(x, coords, edge_index, dist_rbf)
        graph = global_mean_pool(node, data.batch)
        out = self.regression(graph)
        return out