import torch
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

def get_score(label, predict):
    RP = pearsonr(label,predict)[0]
    RS = spearmanr(label,predict)[0]
    MSE = mean_squared_error(label,predict)
    RMSE = MSE**0.5
    return RP, RS, RMSE

def get_train_test_split(dataset, test_size):
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

