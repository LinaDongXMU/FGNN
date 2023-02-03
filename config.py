from yacs.config import CfgNode as CN

def get_config(config):
    config = CN()
    config.cache = ''
    config.random_seed = 1024
    config.epoch = 300
    config.device = 'cpu'

    config.model = CN()
    config.model.node_dim = 36
    config.model.signnet.in_dim = 256
    config.model.signnet.hidden_dim = 256
    config.model.signnet.out_dim = 128
    config.model.signnet.edge_dim = 10

    config.model.attentive.hidden_dim = 256
    config.model.attentive.out_dim = 128
    config.model.attentive.edge_dim = 10
    config.model.attentive.num_layers = 3
    config.model.attentive.num_timestpes = 3

    config.model.regression.in_dim = 128
    config.model.regression.hidden_dim_1 = 1024
    config.model.regression.hidden_dim_2 = 512
    config.model.regression.out_dim = 1
    config.model.regression.dropout = 0.1

    config.dataset = CN()
    config.dataset.path = './data/pdbbind2016_train.pkl'
    config.dataset.pos_enc_dim = 256
    config.dataset.enc_type = 'sym'
    config.dataset.batch_size = 64
    config.dataset.num_workers = 4

    config.scheduler = CN()
    config.scheduler.lr = 0.01
    config.scheduler.mode = 'min'
    config.scheduler.factor = 0.5
    config.scheduler.cooldown = 30
    config.scheduler.min_lr = 1e-6

    config.inference = CN()
    config.inference.train_path = './data/pdbbind2016_train.pkl',
    config.inference.test_path = './data/pdbbind2016_test.pkl',
    return config

config = get_config(CN())
config.freeze()

if __name__ == "__main__":
    print(config)