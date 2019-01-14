import numpy as np
import tensorflow as tf
from src.config import config

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            action_nums,
            feature_nums,
            learning_rate=0.01,
            reward_decay=1,
            output_graph=False,
    ):
        self.action_nums = action_nums
        self.feature_nums = feature_nums
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_states, self.ep_as, self.ep_rs = [], [], [] # 状态，动作，奖励，在一轮训练后存储

        self.build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        with tf.name_scope('inputs'):
            self.tf_states = tf.placeholder(tf.float32, [None, self.feature_nums], name="states")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="action_num")
            # (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="action_value") # 选择此动作的价值-log(act_prob)*v

            # 全连接层第1层
            layer = tf.layers.dense(
                inputs = self.tf_states,
                units = config['neuron_nums'],
                activation = tf.nn.tanh,
                kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'fc1'
            )

            # 全连接层第2层
            all_act = tf.layers.dense(
                inputs = layer,
                units = self.action_nums,
                activation = None,
                kernel_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.3),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'fc2'
            )

            self.all_act_prob = tf.nn.softmax(all_act, name = 'act_prob') # 评价每一个动作选择出现的概率

            with tf.name_scope('loss'):
                # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
                # tf.one_hot在这里是用在分类中，表示对应action的位置
                neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.action_nums), axis=1)
                loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    # 依据概率来选择动作，本身具有随机性
    def choose_action(self, state):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_states: state[np.newaxis, :]})
        # np.random.choice([], p=[])按照一定的概率分布来选择动作
        # .ravel()和.flatten()函数用法相同，区别在于是用了ravel()的数组，如果被修改，那么原数组也会被修改；flatten()不会
        action = np.random.choice(range(1, prob_weights.shape[1]+1), p=prob_weights.ravel())
        return action

    # 储存一回合的s,a,r；因为每回合训练
    def store_transition(self, s, a, r):
        self.ep_states.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # 对每一回合的奖励，进行折扣计算以及归一化
        discounted_ep_rs_norm = self.discount_and_norm_rewards()

        # 一轮训练一次
        self.sess.run(self.train_op, feed_dict={
            self.tf_states: np.vstack(self.ep_states), # shape=[None, states_nums]
            self.tf_acts: np.array(self.ep_as), # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm, # shape=[None, ]
        })

        # 训练完后清除训练数据，开始下一轮
        self.ep_states, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

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
        discounted_ep_rs -= np.mean(discounted_ep_rs) # 均值
        discounted_ep_rs /= np.std(discounted_ep_rs) # 方差
        return discounted_ep_rs


