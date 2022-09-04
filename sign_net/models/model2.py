import torch.nn as nn
from torch_geometric.nn import global_mean_pool, JumpingKnowledge
from layers import EGNNLayer, RegressionLayer, SignNetLayer
from utils import *


class DrugNet(nn.Module):
    def __init__(self, node_dim, hidden_dim, out_dim, edge_dim, num_layer):
        super().__init__()
        
        self.gnn = nn.ModuleList()
        for _ in range(num_layer):
            self.gnn.append(EGNNLayer(node_dim, hidden_dim, edge_dim=edge_dim))
            
        self.jk = JumpingKnowledge(mode='lstm', channels=node_dim, num_layers=num_layer)
        self.node_encoder = nn.Linear(node_dim, out_dim)
        self.regression = RegressionLayer(out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, node, edge_index, coords, dist_rbf, batch):
        node_lst = [node]
        for i, layer in enumerate(self.gnn):
            node, coords = layer(node, coords, edge_index, dist_rbf)
            node_lst.append(node)
        node = self.jk(node_lst)
        node = self.node_encoder(node)
        graph = global_mean_pool(node, batch)
        out = self.regression(graph)
        return out