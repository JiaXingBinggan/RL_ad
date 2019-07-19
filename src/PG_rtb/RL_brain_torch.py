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
        self.fc2 = nn.Linear(config['neuron_nums'], action_numbers)
        self.fc2.weight.data.normal_(0, 0.1)  # 全连接隐层 2 的参数初始化

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        y = self.fc2(x)
        actions_value = F.softmax(y, dim=1)
        return actions_value

def store_para(Net, model_name):
    torch.save(Net.state_dict(), 'Model/PG' + model_name + '_model_params.pth')

class PolicyGradient:
    def __init__(
            self,
            action_nums,
            feature_nums,
            learning_rate=0.01,
            reward_decay=1,
    ):
        self.action_nums = action_nums
        self.feature_nums = feature_nums
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_states, self.ep_as, self.ep_rs = [], [], [] # 状态，动作，奖励，在一轮训练后存储

        self.policy_net = Net(self.feature_nums, self.action_nums).cuda()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def loss_func(self, all_act_prob, acts, vt):
        neg_log_prob = torch.sum(-torch.log(all_act_prob.gather(1, acts - 1))).cuda()
        loss = torch.mean(torch.mul(neg_log_prob, vt)).cuda()
        return loss

    # 依据概率来选择动作，本身具有随机性
    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()
        prob_weights = self.policy_net.forward(state).detach().cpu().numpy()
        action = np.random.choice(range(1, prob_weights.shape[1]+1), p=prob_weights.ravel())
        return action

    # 储存一回合的s,a,r；因为每回合训练
    def store_transition(self, s, a, r):
        self.ep_states.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # 对每一轮的奖励值进行累计折扣及归一化处理
    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float64)
        running_add = 0
        # reversed 函数返回一个反转的迭代器。
        # 计算折扣后的 reward
        # 公式： E = r1 + r2 * gamma + r3 * gamma * gamma + r4 * gamma * gamma * gamma ...
        for i in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[i]
            discounted_ep_rs[i] = running_add

        # 归一化处理
        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 方差
        return discounted_ep_rs

    def learn(self):
        # 对每一回合的奖励，进行折扣计算以及归一化
        discounted_ep_rs_norm = self.discount_and_norm_rewards()

        states = torch.FloatTensor(self.ep_states).cuda()
        acts = torch.unsqueeze(torch.LongTensor(self.ep_as), 1).cuda()
        vt = torch.FloatTensor(discounted_ep_rs_norm).cuda()

        all_act_probs = self.policy_net(states)

        loss = self.loss_func(all_act_probs, acts, vt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 训练完后清除训练数据，开始下一轮
        self.ep_states, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    # 只存储获得最优收益（点击）那一轮的参数
    def para_store_iter(self, test_results):
        max = 0
        if len(test_results) >= 3:
            for i in range(len(test_results)):
                if i == 0:
                    max = test_results[i]
                elif i != len(test_results) - 1:
                    if test_results[i] > test_results[i - 1] and test_results[i] > test_results[i + 1]:
                        if max < test_results[i]:
                            max = test_results[i]
                else:
                    if test_results[i] > max:
                        max = test_results[i]
        return max