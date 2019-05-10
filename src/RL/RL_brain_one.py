import numpy as np
import pandas as pd
import tensorflow as tf
from src.config import config

np.random.seed(1)
tf.set_random_seed(1)

# 定义DeepQNetwork
class DQN:
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
        # e_greedy_increment = , # ε的增量，0-0.9/训练轮次
        out_graph = False,
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
        self.epsilon_increment = None  # epsilon 的增量
        self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + 2)) # 状态的特征数*2加上动作和奖励

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.build_net()
        # 将target_net神经网络的参数替换为eval_net神经网络的参数
        t_params = tf.get_collection('target_net_params') # 提取target_net的神经网络参数
        e_params = tf.get_collection('eval_net_params') # 提取eval_net的神经网络参数
        # 利用tf.assign函数更新target_net参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config['GPU_fraction']) # 分配GPU
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # 是否输出tensorboard文件
        if out_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = [] # 记录所有的cost变化，plot画出


    def store_para(self, model_name):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sess, 'model/1.0_DQN' + model_name + '_model.ckpt')

    def build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.feature_numbers], 'state') # 用于获取状态
        self.q_target = tf.placeholder(tf.float32, [None, self.action_numbers], name='Q_target') # 用于获取目标网络的q值

        w_initializer = tf.random_normal_initializer(0., 0.3) # 权值参数初始化
        b_initializer = tf.constant_initializer(0.1) # 偏置参数初始化

        # 第一层网络的神经元个数，第二层神经元的个数为动作数组的个数
        neuron_numbers = config['neuron_nums']

        # 创建训练神经网络eval_net
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # eval_net的第一层
            with tf.variable_scope('e_l1'):
                w1 = tf.get_variable('w1', [self.feature_numbers, neuron_numbers],
                                     initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, neuron_numbers],
                                     initializer=b_initializer, collections=c_names)
                l1_act = tf.nn.relu(tf.matmul(self.state, w1) + b1) # 第一层的激活函数值

            # eval_net的第二层
            with tf.variable_scope('e_l2'):
                w2 = tf.get_variable('w2', [neuron_numbers, self.action_numbers],
                                     initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.action_numbers],
                                     initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1_act, w2) + b2

        # 创建目标神经网络target_net
        self.state_ = tf.placeholder(tf.float32, [None, self.feature_numbers], name='state_') # 用于获取到下一个状态
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net的第一层
            with tf.variable_scope('target_l1'):
                w1 = tf.get_variable('w1', [self.feature_numbers, neuron_numbers],
                                     initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, neuron_numbers],
                                     initializer=b_initializer, collections=c_names)
                l1_act = tf.nn.relu(tf.matmul(self.state_, w1) + b1)

            # target_net的第二层
            with tf.variable_scope('target_l2'):
                w2 = tf.get_variable('w2', [neuron_numbers, self.action_numbers],
                                     initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.action_numbers],
                                     initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1_act, w2) + b2

        with tf.variable_scope('loss'):  # 求误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):  # 梯度下降
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    # 经验池存储，s-state, a-action, r-reward, s_-state_
    def store_transition(self, s, a, r, s_):
        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录一条[s, a, r, s_]记录
        transition = np.hstack((s, [a, r], s_))

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition # 替换
        self.memory_counter += 1

    # 重置epsilon
    def reset_epsilon(self, e_greedy):
        self.epsilon = e_greedy

    # 选择动作
    def choose_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = np.array(state)[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # 让 eval_net 神经网络生成所有 action 的值, 并选择值最大的 action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: state})
            action = self.action_space[np.argmax(actions_value)] # 选择q_eval值最大的那个动作
            mark = '最优'
        else:
            index = np.random.randint(0, self.action_numbers)
            action = self.action_space[index] # 随机选择动作
            mark = '随机'
        return action, mark

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = np.array(state)[np.newaxis, :]
        # 让 target_net 神经网络生成所有 action 的值, 并选择值最大的 action
        actions_value = self.sess.run(self.q_eval, feed_dict={self.state: state})
        action = self.action_space[np.argmax(actions_value)]  # 选择q_eval值最大的那个动作
        return action

    # 定义DQN的学习过程
    def learn(self):
        # 检查是否达到了替换target_net参数的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print(('\n目标网络参数已经更新\n'))

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
        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.state_: batch_memory[:, -self.feature_numbers:],
                                                  self.state: batch_memory[:, :self.feature_numbers]})
        # 将q_eval拷贝至q_target
        # 下述代码的描述见https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32) # batch数据的序号
        eval_act_array = batch_memory[:, self.feature_numbers] # 动作集合
        eval_act_index = [int(act)-1 for act in eval_act_array] # 获取对应动作在动作空间的的下标

        # eval_act_index = [int(act*100) for act in eval_act_array] # 如果是按“分”为计量单位，则应乘以100

        reward = batch_memory[:, self.feature_numbers + 1] # 奖励集合

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1) # 反向传递更新先前选择的东动作值

        # 训练eval_net
        _, self.cost = self.sess.run([self.train_step, self.loss], feed_dict={self.state: batch_memory[:, :self.feature_numbers],
                                                                              self.q_target: q_target})
        self.cost_his.append(self.cost) # 记录cost误差

        self.learn_step_counter += 1

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