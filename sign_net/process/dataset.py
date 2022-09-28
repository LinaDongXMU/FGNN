import functools
import operator
import pickle
import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data = self._get_graph(data_path)   # 这里初始化时调用数据处理方法对数据进行处理
        self.transform = transform
        
    def _get_graph(self, data_path):   # 这里是我们根据具体任务对数据进行处理
        file = pickle.load(open(data_path, 'rb'), encoding='utf-8')
        data_list = []
        for i, data in tqdm(enumerate(file), desc='Processing File', total=len(file), colour='green'): 
            data = functools.reduce(operator.concat, data)
            try:
                name = data[0]
                node_feature = data[1][0]
                coords = data[1][1]
                edge_index = data[1][2]
                edge_feature = data[1][3]
                dist_rbf = data[1][4]
                label = data[2]
                data = Data(x = torch.from_numpy(node_feature).float(), 
                            y = torch.tensor(label).float(),
                            coords = torch.from_numpy(coords).float(),
                            edge_index = torch.from_numpy(edge_index), 
                            edge_attr = torch.from_numpy(edge_feature).float(),
                            dist_rbf = dist_rbf.float(),
                            id = name)
                data_list.append(data)
            except:
                pass
        return data_list

    def __getitem__(self, index):   # 实现通过索引返回数据
        return self.data[index] if self.transform == None else self.transform(self.data[index])
    
    def __len__(self):   # 返回整个数据集的长度
        return len(self.data)
    
    
class PDBDataset(InMemoryDataset):
    def __init__(self, root: str = None, file_name: str = None, test: bool = False, pre_transform: Callable = None):
        self.file_name = file_name
        self.test = test
        super().__init__(root, pre_transform=pre_transform)
        self.data = torch.load(self.processed_paths[0])
    
    def process(self):
        path = os.path.join(self.root, self.file_name)
        file = pickle.load(open(path, 'rb'), encoding='utf-8')
        data_list = []
        for i, data in tqdm(enumerate(file), desc='Processing File', total=len(file), colour='green'): 
            data = functools.reduce(operator.concat, data)
            try:
                name = data[0]
                node_feature = data[1][0]
                coords = data[1][1]
                edge_index = data[1][2]
                edge_feature = data[1][3]
                dist_rbf = data[1][4]
                label = data[2]
                data = Data(x = torch.from_numpy(node_feature).float(), 
                            y = torch.tensor(label).float(),
                            coords = torch.from_numpy(coords).float(),
                            edge_index = torch.from_numpy(edge_index), 
                            edge_attr = torch.from_numpy(edge_feature).float(),
                            dist_rbf = dist_rbf.float(),
                            id = name)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
            except:
                pass
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return self.file_name
    
    @property
    def processed_file_names(self):
        if self.test:
            return 'pdb_test.pt'
        else:
            return 'pdb_train.pt'
