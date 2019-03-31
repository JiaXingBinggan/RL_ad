import numpy as np
import tensorflow as tf
from src.config import config

np.random.seed(1)
tf.set_random_seed(1)

class RewardNet:
    def __init__(
        self,
        action_space,
        reward_numbers,
        real_reward, # 真实奖励
        feature_numbers,
        learning_rate = 0.01,
        memory_size = 500,
        batch_size = 32,
        out_graph = False
    ):
        self.action_space = action_space
        self.reward_numbers = reward_numbers
        self.real_reward = real_reward
        self.feature_numbers = feature_numbers
        self.lr = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = np.zeros((self.memory_size, self.feature_numbers * 2 + 2))

        # 创建reward_net
        self.build_net()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_faction=config['GPU_fraction'])
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        if out_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        self.state = tf.placeholder(tf.float32, [None, self.feature_numbers], 'state')

        w_initializer = tf.random_normal_initializer(0, 0.3)
        b_initializer = tf.constant_initializer(0.1)

        neuron_numbers = config['neuron_nums']

        with tf.name_scope('reward_net'):
            c_names= ['reward_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('r_l1'):
                w1 = tf.get_variable('w1', [self.feature_numbers, neuron_numbers],
                                     initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, neuron_numbers],
                                     initializer=b_initializer, collections=c_names)
                l1_act = tf.nn.relu(tf.matmul(self.state, w1) + b1)

            with tf.variable_scope('r_l2'):
                w2 = tf.get_variable('w2', [neuron_numbers, self.reward_numbers],
                                     initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.reward_numbers],
                                     initializer=b_initializer, collections=c_names)
                self.model_reward = tf.matmul(l1_act, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.model_reward, self.real_reward))

        with tf.variable_scope('train'):
            self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)


    def store_transition(self, s, s_, ):