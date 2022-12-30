import pickle                                                                                          
                                                                                                       
import numpy as np                                                                                     
import torch                                                                                           
from torch.utils.data import Dataset                                                                   
from torch_geometric.data import Data                                                                  
from tqdm import tqdm                                                                                  
                                                                                                       
                                                                                                       
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