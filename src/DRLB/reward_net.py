import numpy as np
import tensorflow as tf
from .config import config

np.random.seed(1)
tf.set_random_seed(1)

class RewardNet:
    def __init__(
        self,
        action_space,
        reward_numbers,
        feature_numbers,
        learning_rate = 0.01,
        memory_size = 500,
        batch_size = 32,
        out_graph = False
    ):
        self.action_space = action_space
        self.reward_numbers = reward_numbers
        self.feature_numbers = feature_numbers
        self.lr = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        # 将经验池<状态-动作-累积奖励>中的转换组初始化为0
        self.memory_S = np.zeros((self.memory_size, self.feature_numbers + 2))

        # 将经验池<状态-动作-累积奖励中最大>中的转换组初始化为0
        self.memory_D2 = np.zeros((self.memory_size, self.feature_numbers + 2))

        # 创建reward_net
        self.build_net()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config['GPU_fraction'])
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        if out_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.feature_numbers], 'state')

        w_initializer = tf.random_normal_initializer(0, 0.3)
        b_initializer = tf.constant_initializer(0.1)

        neuron_numbers_1 = 100
        neuron_numbers_2 = 100

        with tf.name_scope('reward_net'):
            c_names= ['reward_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('r_l1'):
                w1 = tf.get_variable('w1', [self.feature_numbers, neuron_numbers_1],
                                     initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, neuron_numbers_1],
                                     initializer=b_initializer, collections=c_names)
                l1_act = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope('r_l2'):
                w2 = tf.get_variable('w2', [neuron_numbers_1, neuron_numbers_2],
                                     initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, neuron_numbers_1],
                                     initializer=b_initializer, collections=c_names)
                self.l2_act = tf.nn.relu(tf.matmul(l1_act, w2) + b2)

            with tf.variable_scope('r_l3'):
                w3 = tf.get_variable('w3', [neuron_numbers_2, self.reward_numbers],
                                     initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [self.reward_numbers],
                                     initializer=b_initializer, collections=c_names)
                self.model_reward = tf.matmul(self.l2_act, w3) + b3

        self.real_reward = tf.placeholder(tf.float32, [None, ], 'real_reward')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.model_reward, self.real_reward))

        with tf.variable_scope('train'):
            self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def return_model_reward(self, state):
        # 统一 observation 的 shape (1, size_of_observation)
        state = np.array(state).reshape(1, -1)
        model_reward = self.sess.run(self.model_reward, feed_dict={self.state: state})
        return model_reward

    def store_state_action_pair(self, s, a, model_reward):
        if not hasattr(self, 'memory_S_counter'):
            self.memory_S_counter = 0

        # 记录一条[s,a, m_r]记录
        state_action_pair = np.hstack((s, a, model_reward))

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_S_counter % self.memory_size
        self.memory_S[index, :] = state_action_pair
        self.memory_S_counter += 1

    def store_state_action_reward(self, direct_reward):
        if not hasattr(self, 'memory_D2_counter'):
            self.memory_D2_counter = 0

        for i, memory_s in enumerate(self.memory_S):
            rtn_m = max(self.memory_S[i, -1], direct_reward)
            state_action_rtn = np.hstack((self.memory_S[i, :self.feature_numbers+1], rtn_m))
            index = self.memory_D2_counter % self.memory_size
            self.memory_D2[index, :] = state_action_rtn
            self.memory_D2_counter += 1

    def learn(self):
        sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)

        batch_memory = self.memory_D2[sample_index, :]

        _, self.cost = self.sess.run([self.train_step, self.loss], feed_dict={
            self.state: batch_memory[:, :self.feature_numbers], self.real_reward: batch_memory[:, self.feature_numbers+1]})








