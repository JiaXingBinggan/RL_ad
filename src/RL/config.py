'''
定义一些超参数
'''
import pandas as pd
import numpy as np

config = {
    'e_greedy': 1,
    'learning_rate': 0.1,
    'pg_learning_rate': 1e-4,
    'reward_decay': 1,
    'feature_num': 153,
    'data_pctr_index': 154,
    'data_hour_index': 153, # 17:train-fm
    'data_clk_index': 151, # 15:train-fm
    'data_marketprice_index': 152, # 16:train-fm
    'data_feature_index': 151, # 15:train-fm
    'state_feature_num': 151,
    'train_date': str(20130606), # sample 328481 328 22067108
    'test_date': str(20130607), # sample 307176 307 19441889
    'train_budget': 30096630, # 22067108
    'train_auc_num': 448164, # 155444, 127594, 173710
    'test_budget': 30228554, # 14560732
    'test_auc_num': 478109, # 68244
    'budget_para': [1/1],
    'train_episodes': 500,
    'neuron_nums': 100,
    'GPU_fraction': 1,
    'relace_target_iter': 1000,
    'memory_size': 1000000,
    'batch_size': 32, # GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128...时往往要比设置为整10、整100的倍数时表现更优
}