import numpy as np
import pandas as pd
import tensorflow as tf
from src.config import config

np.random.seed(1)
tf.set_random_seed(1)

# 定义Double DeepQNetwork
class DoubleDQNForTest:
    def __init__(
        self,
        action_space, # 动作空间
        action_numbers, # 动作的数量
        feature_numbers, # 状态的特征数量
        out_graph=False,
    ):
        self.action_space = action_space
        self.action_numbers = action_numbers # 动作的具体数值？[0,0.01,...,budget]
        self.feature_numbers = feature_numbers

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config['GPU_fraction'])  # 分配GPU
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.build_net()

        # 是否输出tensorboard文件
        if out_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def restore_para(self):
        saver = tf.train.import_meta_graph('Model/DDQN_model.ckpt.meta')
        saver.restore(self.sess, 'Model/DDQN_model.ckpt')
        pretrain_graph = tf.get_default_graph()
        self.w1 = self.sess.run(pretrain_graph.get_tensor_by_name('eval_net/e_l1/w1:0'))
        self.b1 = self.sess.run(pretrain_graph.get_tensor_by_name('eval_net/e_l1/b1:0'))
        self.w2 = self.sess.run(pretrain_graph.get_tensor_by_name('eval_net/e_l2/w2:0'))
        self.b2 = self.sess.run(pretrain_graph.get_tensor_by_name('eval_net/e_l2/b2:0'))

    def build_net(self):
        self.restore_para()
        self.state = tf.placeholder(tf.float32, [None, self.feature_numbers], 'state') # 用于获取状态

        # 创建训练神经网络eval_net
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # eval_net的第一层
            with tf.variable_scope('e_l1'):
                w1 = self.w1
                b1 = self.b1
                l1_act = tf.nn.relu(tf.matmul(self.state, w1) + b1) # 第一层的激活函数值

            # eval_net的第二层
            with tf.variable_scope('e_l2'):
                w2 = self.w2
                b2 = self.b2
                self.q_eval = tf.matmul(l1_act, w2) + b2

    # 选择最优动作
    def choose_best_action(self, state):
        # 统一 state 的 shape (1, size_of_state)
        state = np.array(state)[np.newaxis, :]

        # 让 target_net 神经网络生成所有 action 的值, 并选择值最大的 action
        actions_value = self.sess.run(self.q_eval, feed_dict={self.state: state})
        action = self.action_space[np.argmax(actions_value)]  # 选择q_eval值最大的那个动作
        return action