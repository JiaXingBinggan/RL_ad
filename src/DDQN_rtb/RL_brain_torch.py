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

def store_para(Net, model_name):
    torch.save(Net.state_dict(), 'Model/DDQN' + model_name + '_model_params.pth')

# 定义Double DeepQNetwork
class DoubleDQN:
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
        self.epsilon_increment = e_greedy / config['train_episodes']  # epsilon 的增量
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + 2))  # 状态的特征数*2加上动作和奖励

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net, self.target_net = Net(self.feature_numbers, self.action_numbers).cuda(), Net(
            self.feature_numbers, self.action_numbers).cuda()

        # 优化器
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=self.lr, alpha=0.9)
        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss().cuda()

    # 经验池存储，s-state, a-action, r-reward, s_-state_
    def store_transition(self, transition):
        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换
        self.memory_counter += 1

    # 重置epsilon
    def reset_epsilon(self, e_greedy):
        self.epsilon = e_greedy

    # 选择动作
    def choose_action(self, state, state_pctr):
        torch.cuda.empty_cache()
        # epsilon增加步长
        belta = 20
        # 当pctr较高时, 增加epsilon使其利用率增高
        current_epsilon = self.epsilon + state_pctr * belta
        l_epsilon = current_epsilon if current_epsilon < self.epsilon_max else self.epsilon_max  # 当前数据使用的epsilon

        # 统一 state 的 shape, torch.unsqueeze()这个函数主要是对数据维度进行扩充
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()

        if np.random.uniform() < l_epsilon:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.eval_net.forward(state)
            # torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor),按维度dim 返回最大值
            # torch.max(a,1) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一行的行索引）
            action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
            action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
            mark = '最优'
        else:
            index = np.random.randint(0, self.action_numbers)
            action = self.action_space[index]  # 随机选择动作
            mark = '随机'
        return action, mark

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = torch.unsqueeze(torch.FloatTensor(state), 0).cuda()

        actions_value = self.eval_net.forward(state)
        action_index = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        action = self.action_space[action_index]  # 选择q_eval值最大的那个动作
        return action

    # 定义DQN的学习过程
    def learn(self):
        # 清除显存缓存
        torch.cuda.empty_cache()

        # 检查是否达到了替换target_net参数的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print(('\n目标网络参数已经更新\n'))
        self.learn_step_counter += 1

        # 训练过程
        # 从memory中随机抽取batch_size的数据
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory[sample_index, :]

        # 获取到q_next（target_net产生）以及q_eval（eval_net产生）
        # 如store_transition函数中存储所示，state存储在[0, feature_numbers-1]的位置（即前feature_numbets）
        # state_存储在[feature_numbers+1，memory_size]（即后feature_numbers的位置）
        b_s = torch.FloatTensor(batch_memory[:, :self.feature_numbers]).cuda()
        b_a = torch.unsqueeze(torch.LongTensor(batch_memory[:, self.feature_numbers].astype(int)), 1).cuda()
        b_r = torch.FloatTensor(batch_memory[:, self.feature_numbers + 1]).cuda()
        b_s_ = torch.FloatTensor(batch_memory[:, -self.feature_numbers:]).cuda()

        # q_eval w.r.t the action in experience
        # b_a - 1的原因是，出价动作最高300，而数组的最大index为299
        q_eval = self.eval_net.forward(b_s).gather(1, b_a - 1)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.target_net.forward(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练
        # 下一状态s的eval_net值
        q_eval_next = self.eval_net.forward(b_s_)
        max_b_a_next = torch.unsqueeze(torch.max(q_eval_next, 1)[1], 1)  # 选择最大Q的动作
        select_q_next = q_next.gather(1, max_b_a_next).detach()

        q_target = b_r.view(self.batch_size, 1) + self.gamma * select_q_next  # shape (batch, 1)

        q_target = q_target

        # 训练eval_net
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def control_epsilon(self):
        # 逐渐增加epsilon，增加行为的利用性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

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