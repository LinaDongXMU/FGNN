import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_undirected
from torch_sparse import SparseTensor
from tqdm import tqdm


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
                                                                                                       
                                                                                                       
class MyDataset(Dataset):                                                                              
    def __init__(self, data_path, transform=None):                                                     
        super().__init__()                                                                             
        self.data = self._get_graph(data_path)            
        self.transform = transform                                                                     
                                                                                                       
    def _get_graph(self, data_path):                             
        file = pickle.load(open(data_path, 'rb'), encoding='utf-8')                                    
        data_list = []                                                                                 
        for i, data in tqdm(enumerate(file), desc='Processing File', total=len(file), colour='green'): 
            try:                                                                                       
                data = np.ravel(data)                                                                  
                name = data[0]                                                                         
                node_feature = data[1][0]                                                              
                coords = data[1][1]                                                                    
                edge_index = data[1][2]                                                                
                edge_feature = data[1][3]                                                              
                dist_rbf = data[1][4]                                                                  
                label = data[2]                                                                        
                data = Data(x = torch.from_numpy(node_feature),                                        
                            y = torch.tensor(label),                                                   
                            coords = torch.from_numpy(coords),                                         
                            edge_index = torch.from_numpy(edge_index),                                 
                            edge_attr = torch.from_numpy(edge_feature),                                
                            dist_rbf = dist_rbf,                                                       
                            id = name)                                                                 
                data_list.append(data)                                                                 
            except:                                                                                    
                print(f'None Value index is: {i}')                                                     
        return data_list                                                                               
                                                                                                       
    def __getitem__(self, index):                                                
        return self.data[index] if self.transform == None else self.transform(self.data[index])        
                                                                                                       
    def __len__(self):                                                           
        return len(self.data)                                                                          