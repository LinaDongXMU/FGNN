import torch
from scipy import sparse
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch_geometric.utils import get_laplacian, to_undirected, to_scipy_sparse_matrix, degree
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


# The needed pretransform to save result of EVD
class PETransform(object): 
    def __init__(self, pos_enc_dim, enc_type='lap'):
        super().__init__()
        assert enc_type.lower() in ['rw', 'sym'], 'position encoding type error'
        
        self.pos_enc_dim = pos_enc_dim
        self.enc_type = enc_type
        
    def __call__(self, data):
        n = data.num_nodes
        EigVal, EigVec = self._position_encoding(data, self.enc_type)
        position_encoding = EigVec[:,1:self.pos_enc_dim+1]
        
        if n <= self.pos_enc_dim:
            position_encoding = F.pad(position_encoding, (0, self.pos_enc_dim - n + 1), value=float('0'))
            
        data.pos_enc = position_encoding
        return data

    def _position_encoding(self, data, norm=None):
        L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes), 
                              normalization=norm, num_nodes=data.num_nodes)
        L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

        EigVal, EigVec  = torch.linalg.eigh(L)
        return EigVal, EigVec


# class PETransform(object): 
#     def __init__(self, pos_enc_dim, norm=None, enc_type='lap'):
#         super().__init__()
#         assert enc_type.lower() in ['lap', 'rw'], 'position encoding type error'
#         self.pos_enc_dim = pos_enc_dim
#         self.norm = norm
#         self.enc_type = enc_type
        
#     def __call__(self, data):
#         n = data.num_nodes
#         edge_index = data.edge_index
#         if self.enc_type == 'lap':
#             EigVal, EigVec = self._Laplacian(data, self.norm)
#             position_encoding = EigVec[:,1:self.pos_enc_dim+1]
            
#             if n <= self.pos_enc_dim:
#                 position_encoding = F.pad(position_encoding, (0, self.pos_enc_dim - n + 1), value=float('0'))

#         elif self.enc_type == 'rw':
#             position_encoding = self._rw(edge_index, self.pos_enc_dim)
            
#         data.pos_enc = position_encoding
#         return data

#     def _Laplacian(self, data, norm=None):
#         L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes), 
#                               normalization=norm, num_nodes=data.num_nodes)
#         L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

#         EigVal, EigVec  = torch.linalg.eigh(L)
#         return EigVal, EigVec
    
#     def _rw(self, edge_index, pos_enc_dim):
#         """
#             Initializing positional encoding with RWPE
#         """
#         # Geometric diffusion features with Random Walk
#         A = to_scipy_sparse_matrix(edge_index).toarray()
#         Dinv = sparse.diags((degree(edge_index[0]).clip(1) ** -1.0).tolist(), dtype=float) # D^-1
#         RW = A * Dinv  
#         M = RW
        
#         # Iterate
#         nb_pos_enc = pos_enc_dim
#         PE = [torch.tensor(M.diagonal()).float()]
#         M_power = M
#         for _ in range(nb_pos_enc-1):
#             M_power = M_power * M
#             PE.append(torch.tensor(M_power.diagonal()).float())
#         PE = torch.stack(PE,dim=-1)
#         return PE

def to_dense_EVD(eigS, eigV, batch):
    # eigS has the same dimension as batch
    batch_size = int(batch.max()) + 1
    num_nodes = scatter_add(batch.new_ones(eigS.size(0)), batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=eigS.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    eigS_dense = eigS.new_full([batch_size * max_num_nodes], 0) 
    eigS_dense[idx] = eigS
    eigS_dense = eigS_dense.view([batch_size, max_num_nodes])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=eigS.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)
    mask_squared = mask.unsqueeze(2) * mask.unsqueeze(1)
    eigV_dense = eigV.new_full([batch_size * max_num_nodes * max_num_nodes], 0)
    eigV_dense[mask_squared.reshape(-1)] = eigV
    eigV_dense = eigV_dense.view([batch_size, max_num_nodes, max_num_nodes])

    # eigS_dense: B x N_max
    # eigV_dense: B x N_max x N_max
    return eigS_dense, eigV_dense, mask

def to_dense_list_EVD(eigS, eigV, batch):
    eigS_dense, eigV_dense, mask = to_dense_EVD(eigS, eigV, batch)

    nmax = eigV_dense.size(1)
    eigS_dense = eigS_dense.unsqueeze(1).repeat(1, nmax, 1)[mask]
    eigV_dense = eigV_dense[mask]
    # eigS_dense: (N1+N2+...+Nb) x N_max
    # eigV_dense: (N1+N2+...+Nb) x N_max 

    return eigS_dense, eigV_dense

def get_score(label, predict):
    RP = pearsonr(label,predict)[0]
    RS = spearmanr(label,predict)[0]
    MSE = mean_squared_error(label,predict)
    RMSE = MSE**0.5
    return RP, RS, RMSE
