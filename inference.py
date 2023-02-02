import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from process.dataset import *
from utils import *
from config import config

@torch.no_grad()
def valid(data_loader):
    model.eval()
    test_pred = []
    for data in data_loader:
        data = data.to(device)
        predict = model(data).squeeze(-1)
        test_pred.append(predict)
    return torch.cat(test_pred, dim=0).tolist()

if __name__ == "__main__":

    device = config.device
    transform = PETransform(pos_enc_dim=config.dataset.pos_enc_dim,
                            enc_type=config.dataset.enc_type)
    train_dataset = MyDataset(config.inference.train_path, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=64)

    test_dataset = MyDataset(config.inference.test_path, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

    id_train = [i['id'] for i in train_dataset]
    label_train = [i['y'].item() for i in train_dataset]
    dicts_train = {'id':id_train, 'label':label_train}
    df_train = pd.DataFrame(dicts_train)

    id_test = [i['id'] for i in test_dataset]
    label_test = [i['y'].item() for i in test_dataset]
    dicts_test = {'id':id_test, 'label':label_test}
    df_test = pd.DataFrame(dicts_test)

    for i, path in enumerate(os.listdir('./data/cache')):
        model_path = os.path.join('./data/cache', path)
        model = torch.load(model_path)
        predict = valid(train_loader)
        df_train[f'predict_{i+1}'] = predict
    df_train['avg'] = df_train.mean(axis=1)
    df_train.to_csv('./pdbbind2016_train.csv', index=None, encoding='utf-8')

    for i, path in enumerate(os.listdir('./data/cache')):
        model_path = os.path.join('./data/cache', path)
        model = torch.load(model_path)
        predict = valid(test_loader)
        df_test[f'predict_{i+1}'] = predict
    df_test['avg'] = df_test.mean(axis=1)
    df_test.to_csv('./pdbbind2016_test.csv', index=None, encoding='utf-8')
