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

from dataset import MyDataset  # 对这里进行替换，选择量子或经典数据
from model.model import *  # 选择不同的模型
from utils import get_score


class Task():
    def __init__(self, model, dataset, train_idx, valid_idx): # 输入模型，数据集，训练集的index，验证集的index
        self.model = model # 模型
        self.optimizer = optim.Adam(model.parameters(),lr=0.01) # 优化器
        self.warmup_scheduler = warmup.UntunedExponentialWarmup(self.optimizer) # 优化学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, cooldown=40, min_lr=1e-6) # ？？？步长，步数，最小学习率，学习率衰减
        self.criterion = nn.MSELoss() # 损失函数
        
        train_subsampler = SubsetRandomSampler(train_idx) # 训练集采样
        valid_subsampler = SubsetRandomSampler(valid_idx) # 验证集采样

        self.train_loader = DataLoader(dataset, shuffle=False, batch_size=100, sampler=train_subsampler) # 封装训练集
        self.valid_loader = DataLoader(dataset, shuffle=False, batch_size=100, sampler=valid_subsampler) # 封装验证集
    
    def train(self):
        self.model.train() # 模型初始化
        loss_per_epoch_train = 0 # 损失初始化
        label_lst = [] # 标签列表初始化
        train_pred = [] # 预测列表初始化
        for data in self.train_loader: # 遍历封装在train_loader的每条数据
            node_feature = data.x.to(torch.float32).to(device) # 节点特征
            edge_index = data.edge_index.to(torch.long).to(device) # 邻接矩阵
            edge_attr = data.edge_attr.to(torch.float32).to(device) # 边特征
            batch = data.batch.to(torch.long).to(device) # 批次
            label = data.y.to(torch.float32).to(device) # 标签
            
            self.optimizer.zero_grad() # set_to_none=True，初始化优化器权重
            predict = self.model(node_feature, edge_index, edge_attr, batch) # 向模型中传入节点特征，邻接矩阵，边的特征和批次
            label_lst.append(label) # 标签加入标签列表
            train_pred.append(predict) # 预测值加入预测值列表
            loss = self.criterion(predict, label) # 计算损失
            loss.backward() # 损失的反向传播
            self.optimizer.step() # 每个batch更新一次参数
            loss_per_epoch_train += loss.item() # 损失叠加
            
        loss_per_epoch_train = loss_per_epoch_train / len(self.train_loader) # 每个轮次的损失
        return loss_per_epoch_train, torch.cat(train_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist() # 返回每个轮次的损失，合并预测值列表，标签列表
    
    @torch.no_grad()
    def valid(self):
        loss_per_epoch_test = 0
        self.model.eval()
        label_lst = []
        valid_pred = []
        for data in self.valid_loader:
            node_feature = data.x.to(device)
            edge_index = data.edge_index.to(torch.long).to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(torch.long).to(device)
            label = data.y.to(torch.float32).to(device)
            predict = self.model(node_feature, edge_index, edge_attr, batch)
            label_lst.append(label)
            valid_pred.append(predict)
            loss = self.criterion(predict, label)
            loss_per_epoch_test += loss.item()
        # 计算经过一个epoch的训练后再测试集上的损失和精度
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return loss_per_epoch_test, torch.cat(valid_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist() # 和训练的区别就是不需要优化器和误差反向传播

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # gpu
    epochs = 300 # 轮次
    # file=open(r'./data/train_3d_5_5.pkl','rb')
    # dataset = pickle.load(file)
    dataset = MyDataset('./results/pdbbind2016_train.pkl') # 输入数据集

    kf = KFold(n_splits=5, random_state=128, shuffle=True) # 五倍交叉验证
    for kfold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        model = DrugNet(36, 128, 128, 10, 3, 0.1).to(device) # model实例化
        task = Task(model, dataset, train_idx, valid_idx) # task实例化
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
            # ——————————————————————训练————————————————————————
            loss_per_epoch_train, train_predict, train_label = task.train()
            execution_time = time.time() - start_time
            
            # ——————————————————————验证————————————————————————
            loss_per_epoch_valid, valid_predict, valid_label = task.valid()
            time_lst.append(execution_time)
            train_loss_lst.append(loss_per_epoch_train)
            valid_loss_lst.append(loss_per_epoch_valid)
            
            # ——————————————warm_up&lr_scheduler—————————————
            with task.warmup_scheduler.dampening():
                task.scheduler.step(loss_per_epoch_valid)   # 学习率调整
            
            # ——————————————————————相关系数—————————————————————
            train_rp, train_rs, train_rmse = get_score(train_label, train_predict)
            valid_rp, valid_rs, valid_rmse = get_score(valid_label, valid_predict)
            train_rp_lst.append(train_rp)
            train_rs_lst.append(train_rs)
            valid_rp_lst.append(valid_rp)
            valid_rs_lst.append(valid_rs)
            
            # ——————————————————————模型保存—————————————————————
            if (loss_per_epoch_valid < min_loss) and (epoch > 200):
                min_loss = loss_per_epoch_valid
                num += 1
                if num % 2 == 0:
                    torch.save(model, f'./data/cache/gin_{kfold}_1.pkl')
                else:
                    torch.save(model, f'./data/cache/gin_{kfold}_2.pkl')
            if epoch+1 == 300:
                torch.save(model, f'./data/cache/gin_{kfold}_3.pkl')

            # ——————————————————————数据打印—————————————————————
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
