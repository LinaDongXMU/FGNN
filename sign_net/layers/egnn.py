import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class EGNNLayer(MessagePassing):
    def __init__(self, node_dim, hidden_dim, C=1, edge_dim=0, aggr = "add"):
        """E(n) Equivariant Graph Neural Networks from the `"How Powerful are
           Graph Neural Networks?" <https://arxiv.org/abs/2102.09844>`_ paper

        Args:
            node_dim (int): node feature dim
            hidden_dim (int): linear hidden dim
            C (int, optional): _description_. Defaults to 1.
            edge_dim (int, optional): edge feature dim. Defaults to 0.
            aggr (str, optional): aggregate function. Defaults to "add".
        """
        super().__init__(aggr)
        
        self.msg_mlp = nn.Sequential(nn.Linear(2*node_dim+edge_dim+1, hidden_dim),
                                     nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        
        self.coord_update = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          nn.SiLU(),
                                          nn.Linear(hidden_dim, 3))

        self.node_update = nn.Sequential(nn.Linear(node_dim+hidden_dim, hidden_dim),
                                         nn.SiLU(),
                                         nn.Linear(hidden_dim, node_dim))
        self.C = C
        
    def forward(self, node_feature, node_coords, edge_index, edge_feature=None):
        coords_rel = node_coords[edge_index[0]] - node_coords[edge_index[1]]
        dist_rel = torch.sum(coords_rel ** 2, dim=-1, keepdim=True)

        if edge_feature != None:
            edge_feature = torch.cat([dist_rel, edge_feature], dim=-1)
        else:
            edge_feature = dist_rel
            
        msg_agg, coords_agg = self.propagate(edge_index=edge_index, x=node_feature, coords=coords_rel, edge_feature=edge_feature)
        node_update = self.node_update(torch.cat([node_feature, msg_agg], dim=-1))
        coords_update = node_coords + self.C * coords_agg
        return node_update, coords_update
        
    def message(self, x_i, x_j, coords, edge_feature):
        msg = self.msg_mlp(torch.cat([x_i, x_j, edge_feature], dim=-1))
        coords = coords * self.coord_update(msg)
        return msg, coords
    
    def aggregate(self, inputs, index):
        msg, coords = inputs
        msg_agg = scatter(msg, index, dim=0, reduce=self.aggr)
        coords_agg = scatter(coords, index, dim=0, reduce=self.aggr)
        return msg_agg, coords_agg