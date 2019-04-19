import numpy as np
import time
import csv
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
        reward = 0
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

    def step_profit(self, auction_in, action, auction_in_next, pCTR, punishRate, punishNoWinRate, encourageNoClkNoWin):
        alpha = 1e5 # 惩罚程度
        eCPC = 30000
        pRevenue = eCPC * pCTR
        is_win = False

        market_price = float(auction_in[config['data_marketprice_index']])
        if action >= market_price:
            if int(auction_in[config['data_clk_index']]) == 1:
                '''
                调整因子为1-x^2，x为出价与成交价的欧式距离
                当模型出价与市场价越接近，奖励越大
                因为最大化目标为点击，所以无论利润为正或负，都将其设置为绝对值
              '''
                reward =  (- np.square((action - market_price)/action) + 1) * np.abs(pRevenue - market_price)
                # reward = (- np.square((action - market_price) / action) + 1) * 10 * (
                #             pRevenue - market_price)  # 调整因子为1-x^2，x为出价与成交价的欧式距离， 再乘以一个10？以此彰显出价应与成交价相近的值
                print(reward, '1', - np.square((action - market_price)/action) + 1, pRevenue - market_price)
            else:
                reward = -alpha * market_price*punishRate
            self.observation[0] -= float(auction_in[config['data_marketprice_index']])
            self.observation[1] -= 1
            is_win = True
        else:
            if int(auction_in[config['data_clk_index']]) == 1:
                reward = -alpha * pRevenue / punishNoWinRate
            else:
                reward = encourageNoClkNoWin
                print(reward, '2')
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
