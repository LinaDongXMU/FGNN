import json
import os
import pickle
import time

import pytorch_warmup as warmup
import torch
from config import config
from process.dataset import *
from sklearn.model_selection import KFold
from source.model import *
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import *


class Task():
    def __init__(self, model, dataset, train_idx, valid_idx, config):
        self.model = model
        self.device = config.device
        self.optimizer = optim.Adam(model.parameters(),lr=config.scheduler.lr)
        self.warmup_scheduler = warmup.UntunedExponentialWarmup(self.optimizer)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode=config.scheduler.mode,
                                                              factor=config.scheduler.factor,
                                                              cooldown=config.scheduler.cooldown,
                                                              min_lr=config.scheduler.min_lr)
        self.criterion = nn.MSELoss()

        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(dataset,
                                       shuffle=False,
                                       batch_size=config.dataset.batch_size,
                                       sampler=train_subsampler,
                                       num_workers=config.dataset.num_workers)
        self.valid_loader = DataLoader(dataset,
                                       shuffle=False,
                                       batch_size=config.dataset.batch_size,
                                       sampler=valid_subsampler,
                                       num_workers=config.dataset.num_workers)

    def train(self):
        self.model.train()
        loss_per_epoch_train = 0
        label_lst = []
        train_pred = []
        for data in self.train_loader:
            data = data.to(self.device)
            label = data.y.float().to(self.device)
            self.optimizer.zero_grad()
            predict = self.model(data).squeeze(-1)
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
            data = data.to(self.device)
            label = data.y.float().to(self.device)
            predict = self.model(data).squeeze(-1)
            label_lst.append(label)
            valid_pred.append(predict)
            loss = self.criterion(predict, label)
            loss_per_epoch_test += loss.item()

        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return loss_per_epoch_test, torch.cat(valid_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()

def run(config):
    device = config.device
    seed_torch(config.seed)
    transform = PETransform(pos_enc_dim=config.dataset.pos_enc_dim,
                            enc_type=config.dataset.enc_type)

    dataset = MyDataset(data_path=config.dataset.path,
                        transform=transform)

    kf = KFold(n_splits=5, random_state=128, shuffle=True)
    for kfold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        model = DrugNet(config.model).to(device)
        task = Task(model, dataset, train_idx, valid_idx, config)
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
        for epoch in tqdm(range(config.epoch)):
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

if __name__ == "__main__":
    print('='*20)
    print(config)
    print('='*20)
    run(config)


