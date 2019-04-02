'''
定义一些超参数
'''
import pandas as pd
import numpy as np

config = {
    'e_greedy': 1,
    'learning_rate': 0.1,
    'pg_learning_rate': 1e-3,
    'reward_decay': 1,
    'feature_num': 7,
    'state_feature_num': 7,
    'train_date': str(20130606), # sample 328481 328 22067108
    'test_date': str(20130607), # sample 307176 307 19441889
    'train_budget': 5000000, # 22067108
    'train_auc_num': 65000, # 101319
    'test_budget': 5000000, # 14560732
    'test_auc_num': 65000, # 68244
    'budget_para': [1/1],
    'train_episodes': 200,
    'neuron_nums': 100,
    'GPU_fraction': 1,
    'relace_target_iter': 2,
    'memory_size': 100000,
    'batch_size': 32,
}