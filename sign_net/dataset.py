import pickle

import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import distance
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import *


class MyDataset(Dataset):
    def __init__(self, data_path, cut_dist:float, transform=None):
        super().__init__()
        self.cut_dist = cut_dist
        self.data = self._get_graph(data_path)   # 这里初始化时调用数据处理方法对数据进行处理
        self.transform = transform
        
    def _get_graph(self, data_path):   # 这里是我们根据具体任务对数据进行处理
        file = pickle.load(open(data_path, 'rb'), encoding='utf-8')
        id_list = file[0]
        data_list = []
        for i, id in enumerate(id_list):
            coords = file[1][i][1]
            dist_mat = distance.cdist(coords, coords, 'euclidean')
            np.fill_diagonal(dist_mat, np.inf)
            dist_graph_base = dist_mat.copy()
            dist_feat = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1,1)
            dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
            atom_graph = coo_matrix(dist_graph_base)  # (row,col), [data]
            edge_index = torch.tensor(np.array([atom_graph.row, atom_graph.col]), dtype=torch.int64)
            node_feature = torch.from_numpy(file[1][i][2]).float()
            label = file[2][i]
            
            edge_feat_3d = get_3d_feature(edge_index, coords)
            edge_feat_3d[np.isinf(edge_feat_3d)] = np.nan
            edge_feat_3d[np.isnan(edge_feat_3d)] = 0
            edge_feat = torch.tensor(np.hstack((edge_feat_3d, dist_feat)), dtype=torch.float)
            
            data = Data(x = node_feature, 
                        y = label,
                        edge_index = edge_index, 
                        edge_attr = edge_feat,
                        coords = coords,
                        id = id)
            data_list.append(data)
        return data_list

    def __getitem__(self, index):   # 实现通过索引返回数据
        return self.data[index] if self.transform == None else self.transform(self.data[index])
    
    def __len__(self):   # 返回整个数据集的长度
        return len(self.data)
    
if __name__ == "__main__":
    train_path = './data/pdbbind2016_train.pkl'
    print('train_load......')
    train_dataset = MyDataset(train_path, 5.5)
    pickle.dump(train_dataset, open('train_3d_5.5.pkl', 'wb'))
    test_path = './data/pdbbind2016_test.pkl'
    print('test_load......')
    test_dataset = MyDataset(test_path, 5.5)
    pickle.dump(test_dataset, open('test_3d_5.5.pkl', 'wb'))
