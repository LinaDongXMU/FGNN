import torch.nn as nn
from layers import (EGNNLayer, MultiHeadAttentionLayer_2,
                    MultiHeadAttentionLayer_edge_2, RegressionLayer,
                    SignNetLayer, SignNetLayer_Transformer)
from torch_geometric.nn import AttentiveFP, global_mean_pool
from utils import *


# MultiHeadAttention without edge & global_mean_pool
class DrugNet_1(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.out_dim = out_dim
        self.sign_net = SignNetLayer(pos_enc_dim, hidden_dim, node_dim, edge_dim)
        self.attention = MultiHeadAttentionLayer_2(in_dim=node_dim, out_dim=out_dim//8, num_heads=8)
        self.regression = RegressionLayer(out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        # dist_rbf = data.dist_rbf.float()
        pos = self.sign_net(data)   # node_num, 10
        node = self.attention(pos, node_feat, edge_index).view(-1, self.out_dim)
        # graph = self.gnn(x, edge_index, dist_rbf, data.batch)
        graph = global_mean_pool(node, data.batch)
        out = self.regression(graph)
        return out
    
# MultiHeadAttention without edge & AttentiveFP with dist_edge
class DrugNet_2(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.out_dim = out_dim
        self.sign_net = SignNetLayer(pos_enc_dim, hidden_dim, node_dim, edge_dim)
        self.attention = MultiHeadAttentionLayer_2(in_dim=node_dim, out_dim=out_dim//8, num_heads=8)
        self.gnn = AttentiveFP(in_channels=out_dim, hidden_channels=hidden_dim, 
                               out_channels=out_dim, edge_dim=rbf_dim, num_layers=3, num_timesteps=3)
        self.regression = RegressionLayer(out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        dist_rbf = data.dist_rbf
        pos = self.sign_net(data)   # node_num, 10
        node = self.attention(pos, node_feat, edge_index).view(-1, self.out_dim)
        graph = self.gnn(node, edge_index, dist_rbf, data.batch)
        out = self.regression(graph)
        return out
    
# MultiHeadAttention with rbf_edge & global_mean_pool
class DrugNet_3(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.out_dim = out_dim
        self.sign_net = SignNetLayer(pos_enc_dim, hidden_dim, node_dim, edge_dim)
        self.attention = MultiHeadAttentionLayer_edge_2(in_dim=node_dim, out_dim=out_dim//8, num_heads=8, edge_dim=rbf_dim)
        self.regression = RegressionLayer(out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        dist_rbf = data.dist_rbf
        pos = self.sign_net(data)   # node_num, 10
        node, _ = self.attention(pos, node_feat, dist_rbf, edge_index)
        
        node = node.view(-1, self.out_dim)
        
        graph = global_mean_pool(node, data.batch)
        out = self.regression(graph)
        return out
    
# MultiHeadAttention with rbf_edge & AttentiveFP
class DrugNet_4(nn.Module):
    def __init__(self, pos_enc_dim, node_dim, hidden_dim, out_dim, edge_dim, rbf_dim):
        super().__init__()
        self.out_dim = out_dim
        self.sign_net = SignNetLayer(pos_enc_dim, hidden_dim, node_dim, edge_dim)
        self.attention = MultiHeadAttentionLayer_edge_2(in_dim=node_dim, out_dim=out_dim//8, num_heads=8, edge_dim=rbf_dim)
        self.gnn = AttentiveFP(in_channels=out_dim, hidden_channels=hidden_dim, 
                               out_channels=out_dim, edge_dim=out_dim, num_layers=3, num_timesteps=3)
        self.regression = RegressionLayer(out_dim)

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        dist_rbf = data.dist_rbf
        
        pos = self.sign_net(data)   # node_num, 10
        node, edge = self.attention(pos, node_feat, dist_rbf, edge_index)
        
        node = node.view(-1, self.out_dim)
        edge = edge.view(-1, self.out_dim)
        
        graph = self.gnn(node, edge_index, edge, data.batch)
        out = self.regression(graph)
        return out
    
    
class DrugNet_5(nn.Module):
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
