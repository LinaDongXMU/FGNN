import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, cpu_count, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch_geometric.utils import get_laplacian, to_undirected
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from tqdm import tqdm


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

        EigVal, EigVec  = np.linalg.eigh(L)
        EigVal = torch.from_numpy(EigVal)
        EigVec = torch.from_numpy(EigVec)
        return EigVal, EigVec

class EVDTransform(object): 
    def __init__(self, norm=None):
        super().__init__()
        self.norm = norm
        
    def __call__(self, data):
        D, V = EVD_Laplacian(data, self.norm)
        data.eigen_values = D
        data.eigen_vectors = V.reshape(-1) # reshape to 1-d to save 
        return data

def EVD_Laplacian(data, norm=None):
    L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes), 
                          normalization=norm, num_nodes=data.num_nodes)
    L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

    D, V  = torch.linalg.eigh(L)
    return D, V

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

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  """
  Parallel map using joblib.
  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.
  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def get_score(label, predict):
    RP = pearsonr(label,predict)[0]
    RS = spearmanr(label,predict)[0]
    MSE = mean_squared_error(label,predict)
    RMSE = MSE**0.5
    return RP, RS, RMSE
