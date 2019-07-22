import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)

class Net(nn.Module):
    def __init__(self, feature_numbers, reward_numbers):
        super(Net, self).__init__()

        # 第一层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers_1 = 100
        # 第二层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers_2 = 100

        self.fc1 = nn.Linear(feature_numbers, neuron_numbers_1)
        self.fc1.weight.data.normal_(0, 0.1)  # 全连接隐层 1 的参数初始化
        self.fc2 = nn.Linear(neuron_numbers_1, neuron_numbers_2)
        self.fc2.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化
        self.out = nn.Linear(neuron_numbers_1, reward_numbers)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x_1 = self.fc1(input)
        x_1 = F.relu(x_1)
        x_2 = self.fc2(x_1)
        x_2 = F.relu(x_2)
        actions_value = self.out(x_2)
        return actions_value

class RewardNet:
    def __init__(
        self,
        action_space,
        reward_numbers,
        feature_numbers,
        learning_rate = 0.01,
        memory_size = 500,
        batch_size = 32,
    ):
        self.action_space = action_space
        self.reward_numbers = reward_numbers
        self.feature_numbers = feature_numbers
        self.lr = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        if not hasattr(self, 'memory_S_counter'):
            self.memory_S_counter = 0

        if not hasattr(self, 'memory_D2_counter'):
            self.memory_D2_counter = 0

        # 将经验池<状态-动作-累积奖励>中的转换组初始化为0
        self.memory_S = np.zeros((self.memory_size, self.feature_numbers + 2))

        # 将经验池<状态-动作-累积奖励中最大>中的转换组初始化为0
        self.memory_D2 = np.zeros((self.memory_size, self.feature_numbers + 2))

        self.model_reward, self.real_reward = Net(self.feature_numbers, self.reward_numbers).cuda(), Net(self.feature_numbers, self.reward_numbers).cuda()

        # 优化器
        self.optimizer = torch.optim.RMSprop(self.model_reward.parameters(), lr=self.lr, alpha=0.9)
        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss().cuda()

    def return_model_reward(self, state):
        # 统一 observation 的 shape (1, size_of_observation)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()

        model_reward = self.model_reward.forward(state)
        return model_reward

    def store_state_action_pair(self, s, a, model_reward):
        # 记录一条[s,a, m_r]记录
        state_action_pair = np.hstack((s, a, model_reward))

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_S_counter % self.memory_size
        self.memory_S[index, :] = state_action_pair
        self.memory_S_counter += 1

    def store_state_action_reward(self, direct_reward):
        for i, memory_s in enumerate(self.memory_S):
            rtn_m = max(self.memory_S[i, -1], direct_reward)
            state_action_rtn = np.hstack((self.memory_S[i, :self.feature_numbers+1], rtn_m))
            index = self.memory_D2_counter % self.memory_size
            self.memory_D2[index, :] = state_action_rtn
            self.memory_D2_counter += 1

    def learn(self):
        if self.memory_D2_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_D2_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory_D2[sample_index, :]

        states = torch.FloatTensor(batch_memory[:, :self.feature_numbers]).cuda()
        real_reward = torch.unsqueeze(torch.FloatTensor(batch_memory[:, self.feature_numbers + 1]), 1).cuda()

        model_reward = self.model_reward.forward(states)

        loss = self.loss_func(model_reward, real_reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


