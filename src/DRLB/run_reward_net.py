from src.DRLB.reward_net import RewardNet
from src.DRLB.env import AD_env
import numpy as np
import pandas as pd
import copy
import datetime
from src.DRLB.config import config

def bid_func(auc_pCTRS, lamda):
    cpc = 30000
    return auc_pCTRS * cpc / lamda

def run_env(budget, state_array):
    train_data = pd.read_csv('../../data/DRLB/train_DRLB.csv', header=None).drop([0])
    train_data.iloc[:, [0, 2, 3]] = train_data.iloc[:, [0, 2, 3]].astype(int)
    train_data.iloc[:, [1]] = train_data.iloc[:, [1]].astype(float)

    cpc = 30000
    for episode in range(config['train_episodes']):
        V = 0 # 直接奖励值
        for t in range(96):
            state_t = state_array[t][0:8]
            action_t = state_array[t][8]

            auc_t_datas = train_data[train_data.iloc[:, 3].isin([t + 1])]  # t时段的数据
            auc_t_data_pctrs = auc_t_datas.iloc[:, 1].values  # ctrs

            bid_arrays = bid_func(auc_t_data_pctrs, action_t)  # 出价
            win_auc_datas = auc_t_datas[auc_t_datas.iloc[:, 2] <= bid_arrays]  # 赢标的数据
            reward_t = np.sum(win_auc_datas.iloc[:, 1].values * cpc - win_auc_datas.iloc[:, 2].values)

            RewardNet.store_state_action_pair(state_t, action_t, reward_t)







if __name__ == '__main__':
    env = AD_env()
    RewardNet = RewardNet([action for action in np.arange(1, 301)],  # 按照数据集中的“块”计量
             1,153,memory_size=config['memory_size'], batch_size=config['batch_size'],)

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        train_budget, train_auc_numbers = config['train_budget'] * budget_para[i], config['train_auc_num']
        test_budget, test_auc_numbers = config['test_budget'] * budget_para[i], config['test_auc_num']
        run_env(train_budget, train_auc_numbers)