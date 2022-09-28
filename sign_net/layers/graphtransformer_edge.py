import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class MultiHeadAttentionLayer_edge(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, using_bias=False, update_edge_feats=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        
        self.Q = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.K = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.V = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        
    def forward(self, node_feats, edge_feats, edge_index):
        v, z, edge_feat = self.propagate(edge_index=edge_index, x=node_feats, edge_feats=edge_feats)
        
        node_feats = v / z + torch.full_like(z, 1e-6)
        return node_feats, edge_feat
        
    def message(self, x_i, x_j, edge_feats):
        q_i = self.Q(x_i).view(-1, self.num_heads, self.out_dim)
        k_j = self.K(x_j).view(-1, self.num_heads, self.out_dim)
        v_j = self.V(x_j).view(-1, self.num_heads, self.out_dim)
        
        edge_feat_projection = self.edge_feats_projection(edge_feats).view(-1, self.num_heads, self.out_dim)
        att_score = ((q_i * k_j) / np.sqrt(self.out_dim)).clamp(-5.0,5.0)
        edge_feats = torch.exp(att_score * edge_feat_projection).clamp(-5.0,5.0)
        
        msg = edge_feats * v_j
        return msg, edge_feats
        
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        msg, edge_feats = inputs
        
        return [scatter(msg, index, dim=-3, dim_size=dim_size, reduce=self.aggr),
                scatter(edge_feats, index, dim=-3, dim_size=dim_size, reduce=self.aggr),
                edge_feats]

    def update(self, agg_out):
        return agg_out


class GraphTransformerLayer_edge(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer_edge(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)
        
    def forward(self, node_feat, edge_feat, edge_index):
        h_in1 = node_feat # for first residual connection
        e_in1 = edge_feat # for first residual connection
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(node_feat, edge_feat, edge_index)
        
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h # residual connection
            e = e_in1 + e # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h # for second residual connection
        e_in2 = e # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h # residual connection       
            e = e_in2 + e # residual connection  

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)             

        return h, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
        

if __name__ == "__main__":
    node = torch.randn((16, 64))
    edge_index = torch.randint(0, 16, (2, 96))
    edge_feat = torch.randn((96, 64))

    model = GraphTransformerLayer_edge(64, 64, 8, 0.1, True)
    out = model(node, edge_feat, edge_index)
    print(f'node: {out[0].shape}')
    print(f'edge: {out[1].shape}')
