import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data = self._get_graph(data_path)   # 这里初始化时调用数据处理方法对数据进行处理
        self.transform = transform
        
    def _get_graph(self, data_path):   # 这里是我们根据具体任务对数据进行处理
        file = pickle.load(open(data_path, 'rb'), encoding='utf-8')
        total = len(file)
        data_list = []
        for i, data in enumerate(tqdm(file, desc='Processing File', total=total)): 
            try:
                data = np.ravel(data)
                name = data[0]
                node_feature = data[1][0]
                edge_index = data[1][1]
                edge_feature = data[1][2]
                label = data[2]
                data = Data(x = torch.from_numpy(node_feature), 
                            y = label,
                            edge_index = torch.from_numpy(edge_index), 
                            edge_attr = torch.from_numpy(edge_feature),
                            id = name)
                data_list.append(data)
            except:
                print(f'None Value index is: {i}')
        return data_list

    def __getitem__(self, index):   # 实现通过索引返回数据
        return self.data[index] if self.transform == None else self.transform(self.data[index])
    
    def __len__(self):   # 返回整个数据集的长度
        return len(self.data)