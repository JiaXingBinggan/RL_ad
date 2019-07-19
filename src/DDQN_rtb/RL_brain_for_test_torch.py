import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

np.random.seed(1)

class Net(nn.Module):
    def __init__(self, feature_numbers, action_numbers):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_numbers, config['neuron_nums'])
        self.fc1.weight.data.normal_(0, 0.1)  # 全连接隐层 1 的参数初始化
        self.out = nn.Linear(config['neuron_nums'], action_numbers)
        self.out.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# 定义Double DeepQNetwork
class DoubleDQNForTest:
    def __init__(
        self,
        action_space,  # 动作空间
        action_numbers,  # 动作的数量
        feature_numbers,  # 状态的特征数量
        model_name,  # 加载模型名
    ):
        self.action_space = action_space
        self.action_numbers = action_numbers  # 动作的具体数值？[0,0.01,...,budget]
        self.feature_numbers = feature_numbers
        self.model_name = model_name

        # restore params
        self.eval_net = Net(self.feature_numbers, self.action_numbers).cuda()
        self.eval_net.load_state_dict(torch.load('Model/DQN' + self.model_name + '_model_params.pth'))

    # 选择最优动作
    def choose_best_action(self, state):
        torch.cuda.empty_cache()
        # 统一 state 的 shape (1, size_of_state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()
        actions_value = self.eval_net.forward(state)
        action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
        return action