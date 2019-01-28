import numpy as np
import tensorflow as tf
from src.config import config

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradientForTest:
    def __init__(
            self,
            action_nums,
            feature_nums,
            output_graph=False,
    ):
        self.action_nums = action_nums
        self.feature_nums = feature_nums

        self.ep_states, self.ep_as, self.ep_rs = [], [], [] # 状态，动作，奖励，在一轮训练后存储

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config['GPU_fraction'])  # 分配GPU
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.build_net()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def restore_para(self):
        saver = tf.train.import_meta_graph('Model/PG_model.ckpt.meta')
        with tf.Session() as sess:
            saver.restore(sess, 'Model/PG_model.ckpt')
            pretrain_graph = tf.get_default_graph()
            # print(tf.trainable_variables()) # tf.trainable_variables返回的是需要训练的变量列表
            self.fc1_kernel = sess.run(pretrain_graph.get_tensor_by_name('fc1/kernel:0'))
            self.fc1_bias = sess.run(pretrain_graph.get_tensor_by_name('fc1/bias:0'))
            self.fc2_kernel = sess.run(pretrain_graph.get_tensor_by_name('fc2/kernel:0'))
            self.fc2_bias = sess.run(pretrain_graph.get_tensor_by_name('fc2/bias:0'))

    def build_net(self):
        self.restore_para()
        with tf.name_scope('inputs'):
            self.tf_states = tf.placeholder(tf.float32, [None, self.feature_nums], name="states")

            # 全连接层第1层
            layer = tf.nn.tanh(tf.matmul(self.tf_states, self.fc1_kernel) + self.fc1_bias)

            # 全连接层第2层
            all_act = tf.matmul(layer, self.fc2_kernel) + self.fc2_bias

            self.all_act_prob = tf.nn.softmax(all_act, name = 'act_prob') # 评价每一个动作选择出现的概率

    # 依据概率来选择动作，本身具有随机性
    def choose_action(self, state):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_states: state[np.newaxis, :]})
        # np.random.choice([], p=[])按照一定的概率分布来选择动作
        # .ravel()和.flatten()函数用法相同，区别在于是用了ravel()的数组，如果被修改，那么原数组也会被修改；flatten()不会
        action = np.random.choice(range(1, prob_weights.shape[1]+1), p=prob_weights.ravel())
        return action

