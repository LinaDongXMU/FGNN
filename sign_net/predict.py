import os
import torch
import pickle
import pandas as pd
from process.dataset import MyDataset
from torch_geometric.loader import DataLoader

file1=open(r'./results/pdbbind2016_train.pkl','rb')
train_dataset = pickle.load(file1)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=64)
file2=open(r'./results/pdbbind2016_test.pkl','rb')
test_dataset = pickle.load(file2)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

id_train = [i['id'] for i in train_dataset]
label_train = [i['y'] for i in train_dataset]
dicts_train = {'id':id_train, 'label':label_train}
df_train = pd.DataFrame(dicts_train)

id_test = [i['id'] for i in test_dataset]
label_test = [i['y'] for i in test_dataset]
dicts_test = {'id':id_test, 'label':label_test}
df_test = pd.DataFrame(dicts_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def valid(data_loader):
    model.eval()
    test_pred = []
    for data in data_loader:
        node_feature = data.x.to(device)
        edge_index = data.edge_index.to(torch.long).to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(torch.long).to(device)
        predict = model(node_feature, edge_index, edge_attr, batch)
        test_pred.append(predict)
    # 计算经过一个epoch的训练后再测试集上的损失和精度
    return torch.cat(test_pred, dim=0).tolist()

for i, path in enumerate(os.listdir('./data/cache')):
    model_path = os.path.join('./data/cache', path)
    model = torch.load(model_path)
    predict = valid(train_loader)
    df_train[f'predict_{i+1}'] = predict
df_train.to_csv('./pdbbind2016_train.csv', index=None, encoding='utf-8')

for i, path in enumerate(os.listdir('./data/cache')):
    model_path = os.path.join('./data/cache', path)
    model = torch.load(model_path)
    predict = valid(test_loader)
    df_test[f'predict_{i+1}'] = predict
df_test.to_csv('./pdbbind2016_test.csv', index=None, encoding='utf-8')
