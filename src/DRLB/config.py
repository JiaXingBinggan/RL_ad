'''
定义一些超参数
'''
import pandas as pd
import numpy as np

config = {
    'e_greedy': 1,
    'learning_rate': 0.001,
    'pg_learning_rate': 1e-3,
    'reward_decay': 1,
    'feature_num': 7,
    'state_feature_num': 7,
    'train_date': str(20130606), # sample 328481 328 22067108
    'test_date': str(20130607), # sample 307176 307 19441889
    'train_budget': 5000000, # 30096630, 30608307
    'train_auc_num': 437520, # 1448164, 448164, 435900
    'test_budget': 5000000, # 130228554, 30228554, 30231716
    'test_auc_num': 447493, # 478109, 444191
    'init_lamda': 0.000030094, # 1458-1.641490909090909e-5‬, 2.579485714285714e-5‬, 0.000030094‬, 0.000045141; 3386-1.283676923076923e-5‬,0.00002085975‬,0.000027813‬,0.0000417195‬
    'budget_para': [1/1],
    'train_episodes': 1000,
    'neuron_nums': 100,
    'GPU_fraction': 1,
    'relace_target_iter': 100,
    'memory_size': 100000,
    'batch_size': 32,
}