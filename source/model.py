import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP
from layers import *


class RegressionLayer(nn.Module):                                                                                        
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, out_dim, dropout):                                                                                       
        super().__init__()                                                                                               
        self.fc_net = nn.Sequential(                                                                                     
            nn.Linear(in_dim, hidden_dim_1),                                                                                  
            nn.ReLU(),                                                                                                   
            nn.Dropout(dropout),                                                                                             
            nn.Linear(hidden_dim_1, hidden_dim_1),                                                                                       
            nn.ReLU(),                                                                                                   
            nn.Dropout(dropout),                                                                                             
            nn.Linear(hidden_dim_1, hidden_dim_2),                                                                                        
            nn.ReLU(),                                                                                                   
            nn.Dropout(dropout),                                                                                             
            nn.Linear(hidden_dim_2, out_dim)                                                                                            
        )                                                                                                                
                                                                                                                         
    def forward(self, x):                                                                                                
        out = self.fc_net(x)                                                                                             
        return out 
                                                                         

class DrugNet(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        node_dim = config.node_dim
        self.sign_net = SignNetLayer(in_dim = config.signnet.in_dim, 
                                     hidden_dim = config.signnet.hidden_dim, 
                                     out_dim = config.signnet.out_dim, 
                                     edge_dim = config.signnet.edge_dim)
        
        self.gnn = AttentiveFP(in_channels = node_dim+config.signnet.out_dim, 
                               hidden_channels = config.attentive.hidden_dim, 
                               out_channels = config.attentive.out_dim, 
                               edge_dim = config.attentive.edge_dim, 
                               num_layers = config.attentive.num_layers,
                               num_timesteps = config.attentive.num_timestpes)
        
        self.regression = RegressionLayer(in_dim = config.regression.in_dim,
                                          hidden_dim_1 = config.regression.hidden_dim_1,
                                          hidden_dim_2 = config.regression.hidden_dim_2,
                                          out_dim = config.regression.out_dim,
                                          dropout = config.regression.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        node_feat = data.x.float()
        edge_index = data.edge_index
        dist_rbf = data.dist_rbf.float()
        pos = self.sign_net(data)
        x = torch.cat([node_feat, pos], dim=-1)
        batch= data.batch.to(torch.long)
        graph = self.gnn(x, edge_index, dist_rbf, batch)
        out = self.regression(graph)
        return out                                                                