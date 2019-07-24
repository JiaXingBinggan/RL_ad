import numpy as np
import random
from src.config import config

random.seed(1)
class AD_env:
    def __init__(self):
        super(AD_env, self).__init__()
        # self.action_space = [action for action in np.arange(0, 300, 0.01)] # 按照真实货币单位“分”
        self.action_space = [action for action in np.arange(1, 301)] # 按照数据集中的“块”计量
        self.action_numbers = len(self.action_space)
        self.feature_numbers = config['feature_num'] # config['feature_num'] = 1+1+config['state_feature_num']，其中config['state_feature_num']为auction的特征数（隐向量加ctr），第1个1为预算b，第二个为剩余拍卖数量t

    # 创建出价环境
    # 状态要为矩阵形式
    def build_env(self, budget, auction_numbers):
        self.budget = budget
        self.auc_num = auction_numbers # 期望投标数量

        observation = []
        observation.append(budget)
        observation.append(auction_numbers) # 剩余拍卖数量t
        observation[2: config['feature_num']] = [0 for i in range(config['state_feature_num'])] # config['state_feature_num']个特征

        self.observation = observation


    # 重置出价环境
    def reset(self, budget, auction_numbers):
        # self.update()
        self.observation[0] = budget
        self.observation[1] = auction_numbers
        self.observation[2: config['feature_num']] = [0 for i in range(config['state_feature_num'])] # config['state_feature_num']个特征

        return self.observation

    def step(self, auction_in, action, auction_in_next):
        is_win = False
        if action >= float(auction_in[config['data_marketprice_index']]):
            reward = int(auction_in[config['data_clk_index']])
            self.observation[0] -= float(auction_in[config['data_marketprice_index']])
            self.observation[1] -= 1
            is_win = True
        else:
            reward = 0
            self.observation[1] -= 1

        if self.observation[0] <= 0:
            done = True
        elif self.observation[1] <= 0:
            done = True
        else:
            done = False
        observation_ = self.observation
        if len(auction_in_next) == 0:
            auction_in_next = [0 for i in range(0, config['state_feature_num'])]
        observation_[2: config['feature_num']] = auction_in_next

        return observation_, reward, done, is_win

    def step_for_test(self, auction_in, action):
        is_win = False
        if action >= float(auction_in[config['data_marketprice_index']]):
            reward = int(auction_in[config['data_clk_index']])
            self.observation[0] -= float(auction_in[config['data_marketprice_index']])
            self.observation[1] -= 1
            is_win = True
        else:
            reward = 0
            self.observation[1] -= 1

        if self.observation[0] <= 0:
            done = True
        elif self.observation[1] <= 0:
            done = True
        else:
            done = False
        observation_ = self.observation

        return observation_, reward, done, is_win