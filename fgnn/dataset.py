import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = self._get_graph(data_path)   # 这里初始化时调用数据处理方法对数据进行处理
        
    def _get_graph(self, data_path):   # 这里是我们根据具体任务对数据进行处理
        file = pickle.load(open(data_path, 'rb'), encoding='utf-8')
        id_list = file[0]
        data_list = []
        for i, id in enumerate(id_list):
        	
            node_feature = torch.from_numpy(file[1][i][0]).float()
            edge_index = torch.from_numpy(file[1][i][1])
            edge_feat = torch.from_numpy(file[1][i][2])
            
            # laplacian=utils.get_laplacian(edge_index,normalization='sym')
            # L = utils.to_scipy_sparse_matrix(laplacian[0], laplacian[1]).tocsc()
            # EigVal, EigVec = np.linalg.eig(L.toarray())
            # idx = EigVal.argsort() # increasing order
            # EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:,idx])
            # EigVal=torch.from_numpy(EigVal).float().unsqueeze(-1)
            
            label = file[2][i]
            data = Data(x = node_feature,
                        y = label, 
                        edge_index = edge_index, 
                        edge_attr = edge_feat,
                        id = id)
            data_list.append(data)
        return data_list

    def __getitem__(self, index):   # 实现通过索引返回数据
        return self.data[index]
    
    def __len__(self):   # 返回整个数据集的长度
        return len(self.data)
    
if __name__ == "__main__":
    path = './results/pdbbind2016_train.pkl'
    dataset = MyDataset(path)
    train_loader = DataLoader(dataset, batch_size=3, shuffle=True)
    for i in train_loader:
        data = i
        print(data)
        break