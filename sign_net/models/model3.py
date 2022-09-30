import torch.nn as nn
from layers import GraphTransformerLayer_edge, RegressionLayer, SignNetLayer
from torch_geometric.nn import MLP, AttentiveFP, global_mean_pool
from utils import *


class DrugNet_1(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.sign_net = SignNetLayer(pos_enc_dim, hidden_dim, out_dim, edge_dim)
        
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
    
    
class DrugNet_2(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.gnn1 = GraphTransformerLayer_edge(in_dim=node_dim, out_dim=out_dim, num_heads=8, edge_dim=edge_dim, residual=False, use_bias=True)
        
        self.gnn2 = AttentiveFP(in_channels=node_dim, hidden_channels=hidden_dim, 
                               out_channels=out_dim, edge_dim=edge_dim, num_layers=3, num_timesteps=3)
        
        self.embedding1 = MLP([out_dim, hidden_dim, out_dim])
        self.embedding2 = MLP([out_dim, hidden_dim, out_dim])
        
        self.regression = RegressionLayer(4*out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x
        edge_index = data.edge_index
        edge_feat = data.edge_attr
        batch = data.batch
        
        graph1 = global_mean_pool(self.gnn1(node_feat, edge_feat, edge_index)[0], batch)   # node_num, 10
        graph1_e = self.embedding1(graph1)
        
        graph2 = self.gnn2(node_feat, edge_index, edge_feat, batch)
        graph2_e = self.embedding2(graph2)
        
        graph_cat = torch.cat([graph1, graph1_e, graph2, graph2_e], dim=-1)
        out = self.regression(graph_cat)
        return out
