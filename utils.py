import os
import pickle
import random

import numpy as np
import torch
from joblib import Parallel, cpu_count, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def seed_torch(seed=1024):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  """
  Parallel map using joblib.
  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.
  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def random_split(dataset_size, split_ratio=1, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx

def write_pickle(data, output_path, dataset_name):
    train = []
    valid = []
    test = []
    for i in data:
        train.append(i[0])
        valid.append(i[1])
        test.append(i[2])

    with open(os.path.join(output_path, dataset_name + '_train.pkl'), 'wb') as f:
        pickle.dump(train, f)
    with open(os.path.join(output_path, dataset_name + '_val.pkl'), 'wb') as f:
        pickle.dump(valid, f)
    with open(os.path.join(output_path, dataset_name + '_test.pkl'), 'wb') as f:
        pickle.dump(test, f)

def get_score(label, predict):
    RP = pearsonr(label,predict)[0]
    RS = spearmanr(label,predict)[0]
    MSE = mean_squared_error(label,predict)
    RMSE = MSE**0.5
    return RP, RS, RMSE
