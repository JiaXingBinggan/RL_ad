import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

np.random.seed(1)

# 定义DeepQNetwork
class DQN_FOR_TEST:
    def __init__(
        self,
        action_space, # 动作空间
        action_numbers, # 动作的数量
        feature_numbers, # 状态的特征数量
        learning_rate = 0.01, # 学习率
        reward_decay = 1, # 奖励折扣因子,偶发过程为1
        e_greedy = 0.9, # 贪心算法ε
        replace_target_iter = 300, # 每300步替换一次target_net的参数
        memory_size = 500, # 经验池的大小
        batch_size = 32, # 每次更新时从memory里面取多少数据出来，mini-batch
    ):
        self.action_space = action_space
        self.action_numbers = action_numbers # 动作的具体数值？[0,0.01,...,budget]
        self.feature_numbers = feature_numbers
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon_increment = e_greedy/config['train_episodes']  # epsilon 的增量
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + 2)) # 状态的特征数*2加上动作和奖励

        self.restore_para('threshold')
        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net = self.Net(self.feature_numbers, self.action_numbers)

        # 优化器
        self.optimizer = torch.optim.R(self.eval_net.parameters(), lr=self.lr)
        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss()

        self.cost_his = [] # 记录所有的cost变化，plot画出


    def restore_para(self, model_name):
        self.Net = torch.load('Model/DQN' + model_name + '_model.pth')

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.eval_net.forward(state)
        action_index = torch.max(actions_value, 1)[1].data.numpy()[0]
        action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
        return action