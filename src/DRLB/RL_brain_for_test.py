import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config
import os

np.random.seed(1)

class Net(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Net, self).__init__()

        # 第一层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers_1 = 100
        # 第二层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers_2 = 100

        self.fc1 = nn.Linear(feature_numbers, neuron_numbers_1)
        self.fc1.weight.data.normal_(0, 0.1)  # 全连接隐层 1 的参数初始化
        self.fc2 = nn.Linear(neuron_numbers_1, neuron_numbers_2)
        self.fc2.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化
        self.out = nn.Linear(neuron_numbers_1, action_numbers)
        self.out.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化

    def forward(self, input):
        x_1 = self.fc1(input)
        x_1 = F.relu(x_1)
        x_2 = self.fc2(x_1)
        x_2 = F.relu(x_2)
        actions_value = self.out(x_2)
        return actions_value

def store_para(Net):
    torch.save(Net.state_dict(), 'Model/DRLB_model_params.pth')

# 定义DeepQNetwork
class DRLB:
    def __init__(
            self,
            action_space,  # 动作空间
            action_numbers,  # 动作的数量
            feature_numbers,  # 状态的特征数量
            learning_rate=0.01,  # 学习率
            reward_decay=1,  # 奖励折扣因子,偶发过程为1
            e_greedy=0.9,  # 贪心算法ε
            replace_target_iter=300,  # 每300步替换一次target_net的参数
            memory_size=500,  # 经验池的大小
            batch_size=32,  # 每次更新时从memory里面取多少数据出来，mini-batch
    ):
        self.action_space = action_space
        self.action_numbers = action_numbers  # 动作的具体数值？[0,0.01,...,budget]
        self.feature_numbers = feature_numbers
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon = 0.9

        # restore params
        self.eval_net = Net(self.feature_numbers, self.action_numbers).cuda()
        self.eval_net.load_state_dict(torch.load('Model/DRLB_model_params.pth'))

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()

        actions_value = self.eval_net.forward(state)
        action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
        return action