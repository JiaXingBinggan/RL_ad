import numpy as np
import time
import csv
import random

random.seed(1)
class AD_env:
    def __init__(self):
        super(AD_env, self).__init__()
        self.action_space = [action for action in np.arange(0, 300, 0.01)]
        self.action_numbers = len(self.action_space)
        self.feature_numbers = 18 # 18 = 1+1+16，其中11为auction的特征数，第1个1为预算b，第二个为剩余拍卖数量t

    # 创建出价环境
    # 状态要为矩阵形式
    def build_env(self, budget, auction_numbers):
        observation = []
        observation.append(budget)
        observation.append(auction_numbers) # 剩余拍卖数量t
        observation[2: 18] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 16个特征

        self.observation = observation


    # 重置出价环境
    def reset(self, budget, auction_numbers):
        # self.update()
        self.observation[0] = budget
        self.observation[1] = auction_numbers
        self.observation[2: 18] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        return self.observation

    def step(self, auction_in, action):
        reward = 0
        is_win = False
        if action >= float(auction_in[17]):
            reward = int(auction_in[16])
            self.observation[0] -= float(auction_in[17])
            self.observation[1] -= 1
            is_win = True
        else:
            reward = 0
            self.observation[1] -= 1
        observation_ = self.observation

        if self.observation[0] <= 0:
            done = True
        elif self.observation[1] <= 0:
            done = True
        else:
            done = False

        return observation_, reward, done, is_win

    # def render(self):
    #     self.update()
