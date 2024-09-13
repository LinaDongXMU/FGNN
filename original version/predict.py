import os
import torch
import pickle
import pandas as pd
from dataset import MyDataset
from torch_geometric.loader import DataLoader
from utils import *

transform = PETransform(pos_enc_dim=256, enc_type='sym')
train_dataset = MyDataset('./data/pdbbind2016_train.pkl',transform=transform)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=64)

test_dataset = MyDataset('./data/pdbbind2016_test.pkl',transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

id_train = [i['id'] for i in train_dataset]
label_train = [i['y'].item() for i in train_dataset]
dicts_train = {'id':id_train, 'label':label_train}
df_train = pd.DataFrame(dicts_train)

id_test = [i['id'] for i in test_dataset]
label_test = [i['y'].item() for i in test_dataset]
dicts_test = {'id':id_test, 'label':label_test}
df_test = pd.DataFrame(dicts_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def valid(data_loader):
    model.eval()
    test_pred = []
    for data in data_loader:
        data = data.to(device)
        predict = model(data).squeeze(-1)
        test_pred.append(predict)
    return torch.cat(test_pred, dim=0).tolist()

for i, path in enumerate(os.listdir('./data/cache')):
    model_path = os.path.join('./data/cache', path)
    model = torch.load(model_path)
    predict = valid(train_loader)
    df_train[f'predict_{i+1}'] = predict
df_train.eval('pre = (predict_1 + predict_2 + predict_3+predict_4+predict_5+predict_6 + predict_7 + predict_8+predict_9+predict_10+predict_11 + predict_12 + predict_13+predict_14+predict_15)/15' , inplace=True)
df_train.to_csv('./pdbbind2016_train.csv', index=None, encoding='utf-8')

for i, path in enumerate(os.listdir('./data/cache')):
    model_path = os.path.join('./data/cache', path)
    model = torch.load(model_path)
    predict = valid(test_loader)
    df_test[f'predict_{i+1}'] = predict
df_test.eval('pre = (predict_1 + predict_2 + predict_3+predict_4+predict_5+predict_6 + predict_7 + predict_8+predict_9+predict_10+predict_11 + predict_12 + predict_13+predict_14+predict_15)/15' , inplace=True)
df_test.to_csv('./pdbbind2016_test.csv', index=None, encoding='utf-8')
