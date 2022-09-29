import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, using_bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.using_bias = using_bias
        
        self.Q = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.K = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.V = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.Q.reset_parameters()
        self.K.reset_parameters()
        self.V.reset_parameters()
        
    def forward(self, node_feats, edge_index):
        v, z = self.propagate(edge_index=edge_index, x=node_feats)
        
        node_feats = v / z + torch.full_like(z, 1e-6)
        return node_feats
        
    def message(self, x_i, x_j):
        q_i = self.Q(x_i).view(-1, self.num_heads, self.out_dim)
        k_j = self.K(x_j).view(-1, self.num_heads, self.out_dim)
        v_j = self.V(x_j).view(-1, self.num_heads, self.out_dim)
        
        att_score = (q_i * k_j).sum(-1, keepdim=True)
        att_score = torch.exp(att_score / np.sqrt(self.out_dim)).clamp(-5.0,5.0)
        
        msg = v_j * att_score
        return msg, att_score
        
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        msg, att_score = inputs
        
        return [scatter(msg, index, dim=0, dim_size=dim_size, reduce=self.aggr),
                scatter(att_score, index, dim=0, dim_size=dim_size, reduce=self.aggr)]

    def update(self, agg_out):
        return agg_out
    
    
class MultiHeadAttentionLayer_2(MessagePassing):
    def __init__(self, in_dim, out_dim, num_heads, using_bias=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.using_bias = using_bias
        
        self.Q = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.K = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        self.V = nn.Linear(in_dim, out_dim*num_heads, bias=using_bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.Q.reset_parameters()
        self.K.reset_parameters()
        self.V.reset_parameters()
        
    def forward(self, input1, input2, edge_index):
        v, z = self.propagate(edge_index=edge_index, input1=input1, input2=input2)
        
        node_feats = v / z + torch.full_like(z, 1e-6)
        return node_feats
        
    def message(self, input1_i, input2_j):
        q_i = self.Q(input1_i).view(-1, self.num_heads, self.out_dim)
        k_j = self.K(input2_j).view(-1, self.num_heads, self.out_dim)
        v_j = self.V(input2_j).view(-1, self.num_heads, self.out_dim)
        
        att_score = (q_i * k_j).sum(-1, keepdim=True)
        att_score = torch.exp(att_score / np.sqrt(self.out_dim)).clamp(-5.0,5.0)
        
        msg = v_j * att_score
        return msg, att_score
        
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        msg, att_score = inputs
        
        return [scatter(msg, index, dim=0, dim_size=dim_size, reduce=self.aggr),
                scatter(att_score, index, dim=0, dim_size=dim_size, reduce=self.aggr)]

    def update(self, agg_out):
        return agg_out


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual       
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.attention.reset_parameters()
        self.O.reset_parameters()
        self.FFN_layer1.reset_parameters()
        self.FFN_layer2.reset_parameters()
        
    def forward(self, node_feat, edge_index):
        h_in1 = node_feat # for first residual connection
        
        # multi-head attention out
        attn_out = self.attention(node_feat, edge_index)
        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
        

if __name__ == "__main__":
    node_1 = torch.randn((16, 64))
    node_2 = torch.randn((16, 64))
    edge_index = torch.randint(0, 16, (2, 96))
    edge_feat = torch.randn((96, 64))

    model = MultiHeadAttentionLayer_2(64, 128//8, 8)
    out = model(node_1, node_2, edge_index)
    print(f'node: {out.shape}')
    # print(f'edge: {out[1].shape}')
