import json                                                                                                                                                                                       
import os                                                                                                                                                                                         
import time                                                                                                                                                                                       
import pickle                                                                                                                                                                                     
import torch                                                                                                                                                                                      
from sklearn.model_selection import KFold                                                                                                                                                         
from torch import optim                                                                                                                                                                           
from torch.utils.data import SubsetRandomSampler                                                                                                                                                  
from torch_geometric.loader import DataLoader                                                                                                                                                     
from tqdm import tqdm                                                                                                                                                                             
import pytorch_warmup as warmup                                                                                                                                                                   
                                                                                                                                                                                                  
from dataset import MyDataset                                                                                                                         
from model import *                                                                                                                                                        
from utils import get_score                                                                                                                                                                       
                                                                                                                                                                                                  
                                                                                                                                                                                                  
class Task():
    def __init__(self, model, dataset, train_idx, valid_idx):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=0.01)
        self.warmup_scheduler = warmup.UntunedExponentialWarmup(self.optimizer)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, cooldown=30, min_lr=1e-6)
        self.criterion = nn.MSELoss()
        
        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(dataset, shuffle=False, batch_size=64, sampler=train_subsampler, num_workers=4)
        self.valid_loader = DataLoader(dataset, shuffle=False, batch_size=64, sampler=valid_subsampler, num_workers=4)
    
    def train(self):
        self.model.train()
        loss_per_epoch_train = 0
        label_lst = []
        train_pred = []
        for data in self.train_loader:
            data = data.to(device)   # model1
            # node = data.x.float().to(device)     # model2              
            # edge_index = data.edge_index.to(device)   # model2   
            # coords = data.coords.to(device)   # model2   
            # dist_rbf = data.dist_rbf.float().to(device)   # model2   
            # batch = data.batch.to(device)   # model2   
            label = data.y.float().to(device)
            self.optimizer.zero_grad()
            # predict = self.model(node, edge_index, coords, dist_rbf, batch).squeeze(-1)   # model2   
            predict = self.model(data).squeeze(-1)   # model1
            label_lst.append(label)
            train_pred.append(predict)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()         
            loss_per_epoch_train += loss.item()
            
        loss_per_epoch_train = loss_per_epoch_train / len(self.train_loader)
        return loss_per_epoch_train, torch.cat(train_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()
    
    @torch.no_grad()
    def valid(self):
        loss_per_epoch_test = 0
        self.model.eval()
        label_lst = []
        valid_pred = []
        for data in self.valid_loader:
            data = data.to(device)
            # node = data.x.float().to(device)
            # edge_index = data.edge_index.to(device)
            # coords = data.coords.to(device)
            # dist_rbf = data.dist_rbf.float().to(device)
            # batch = data.batch.to(device)
            label = data.y.float().to(device)
            # predict = self.model(node, edge_index, coords, dist_rbf, batch).squeeze(-1)
            predict = self.model(data).squeeze(-1)
            label_lst.append(label)
            valid_pred.append(predict)
            loss = self.criterion(predict, label)
            loss_per_epoch_test += loss.item()
        
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return loss_per_epoch_test, torch.cat(valid_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()

if __name__ == "__main__":
    
    device = 'cuda'
    epochs = 300
    transform = PETransform(pos_enc_dim=256, enc_type='sym')
    print('Loading Data')
    dataset = MyDataset('./data/pdbbind2016_train.pkl', transform=transform)
    print('Loading Done')

    kf = KFold(n_splits=5, random_state=128, shuffle=True)
    for kfold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        # model = DrugNet(node_dim=36, hidden_dim=64, out_dim=128, edge_dim=15, num_layer=3).to(device)   # model2
        model = DrugNet(pos_enc_dim=256, node_dim=36, hidden_dim=256, out_dim=128, edge_dim=10, rbf_dim=15).to(device)   # model                                                                                     
        task = Task(model, dataset, train_idx, valid_idx)                                                                                                                                         
        train_loss_lst = []                                                                                                                                                                       
        valid_loss_lst = []                                                                                                                                                                       
        time_lst = []                                                                                                                                                                             
        train_rp_lst = []                                                                                                                                                                         
        train_rs_lst = []                                                                                                                                                                         
        valid_rp_lst = []                                                                                                                                                                         
        valid_rs_lst = []                                                                                                                                                                         
        num = 0                                                                                                                                                                                   
                                                                                                                                                                                                  
        min_loss = 10.0                                                                                                                                                                           
        start_time =time.time()                                                                                                                                                                   
        for epoch in tqdm(range(epochs)):                                                                                                                                                         
            # ——————————————————————train————————————————————————                                                                                    
            loss_per_epoch_train, train_predict, train_label = task.train()                                                                                                                       
            execution_time = time.time() - start_time                                                                                                                                             
                                                                                                                                                                                                  
            # ——————————————————————valid————————————————————————                                                                                    
            loss_per_epoch_valid, valid_predict, valid_label = task.valid()                                                                                                                       
            time_lst.append(execution_time)                                                                                                                                                       
            train_loss_lst.append(loss_per_epoch_train)                                                                                                                                           
            valid_loss_lst.append(loss_per_epoch_valid)                                                                                                                                           
                                                                                                                                                                                                  
            # ——————————————warm_up&lr_scheduler—————————————                                                                                                          
            with task.warmup_scheduler.dampening():                                                                                                                                               
                task.scheduler.step(loss_per_epoch_valid)                                                                                                                            
                                                                                                                                                                                                  
            # ——————————————————————correlation—————————————————————                                                                                      
            train_rp, train_rs, train_rmse = get_score(train_label, train_predict)                                                                                                                
            valid_rp, valid_rs, valid_rmse = get_score(valid_label, valid_predict)                                                                                                                
            train_rp_lst.append(train_rp)                                                                                                                                                         
            train_rs_lst.append(train_rs)                                                                                                                                                         
            valid_rp_lst.append(valid_rp)                                                                                                                                                         
            valid_rs_lst.append(valid_rs)                                                                                                                                                         
                                                                                                                                                                                                  
            # ——————————————————————save_models—————————————————————                                                                                      
            if (loss_per_epoch_valid < min_loss) and (epoch > 200):                                                                                                                               
                min_loss = loss_per_epoch_valid                                                                                                                                                   
                num += 1                                                                                                                                                                          
                if num % 2 == 0:                                                                                                                                                                  
                    torch.save(model, f'./data/cache/DTI_{kfold}_1.pkl')                                                                                                                        
                else:                                                                                                                                                                             
                    torch.save(model, f'./data/cache/DTI_{kfold}_2.pkl')                                                                                                                        
            if epoch+1 == 300:                                                                                                                                                                    
                torch.save(model, f'./data/cache/DTI_{kfold}_3.pkl')                                                                                                                            
                                                                                                                                                                                                  
            # ——————————————————————print_data—————————————————————                                                                                      
            print(f'kfold: {kfold+1} || epoch: {epoch+1}')                                                                                                                                        
            print(f'train_loss: {loss_per_epoch_train:.3f} || train_rp: {train_rp:.3f} || train_rs: {train_rs:.3f} || train_rmse {train_rmse:.3f}')                                               
            print(f'valid_loss: {loss_per_epoch_valid:.3f} || valid_rp: {valid_rp:.3f} || valid_rs: {valid_rs:.3f} || valid_rmse {valid_rmse:.3f}')                                               
                                                                                                                                                                                                  
        save_path = "./data/data_cache/"                                                                                                                                                               
        if not os.path.exists(save_path):                                                                                                                                                         
            os.mkdir(save_path)                                                                                                                                                                   
        dict = {"train_loss": train_loss_lst, "test_loss": valid_loss_lst, "time": time_lst, "train_rp": train_rp_lst, "train_rs": train_rs_lst, "test_rp": valid_rp_lst, "test_rs": valid_rs_lst}
        with open(save_path + f"train_gin{kfold}.json", "w") as f:                                                                                                                                
            json.dump(dict, f)                                                                                                                                                                    
                                                                                                                                                                                                  
    print('Finished training ')                                                                                                                                                                   
