import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.metrics import log_loss, roc_auc_score

class FM:
    def __init__(
        self,
        latent_dims, # 隐向量维度
        lr, # 学习率
        batch_size,
        feature_length, # 特征长度
    ):
        self.latent_dims = latent_dims
        self.lr = lr
        self.batch_size = batch_size
        self.reg_l1 = tf.contrib.layers.l1_regularizer(0.01)
        self.reg_l2 = tf.contrib.layers.l1_regularizer(0.01)
        self.featuere_length = feature_length

    def add_placeholders(self):
        self.X = tf.placeholder('float64', shape=(None, self.featuere_length))
        self.y = tf.placeholder('float64', shape=(None, 1))
        '''
        为了防止或减轻过拟合,Dropout就是在不同的训练过程中随机扔掉一部分神经元。
        也就是让某个神经元的激活值以一定的概率p，
        让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
       '''
        self.keep_prob = tf.placeholder('float64')

    def inference(self):
        '''
        定义FM框架
        :return: 对每一条数据的预测标签
        '''
        with tf.variable_scope('linaer_layer'):
            self.b = tf.get_variable('bias', shape=[1], initializer=tf.ones_initializer(), dtype=tf.float64, regularizer=self.reg_l2)
            self.w1 = tf.get_variable('w1', shape=[self.featuere_length, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02), regularizer=self.reg_l2,
                                      dtype=tf.float64)
            self.linear_item = tf.add(tf.matmul(self.X, self.w1), self.b)

        # 定义交叉项
        with tf.variable_scope('interaction_layer'):
            self.v = tf.get_variable('v', shape=[self.featuere_length, self.latent_dims],
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02), regularizer=self.reg_l2,
                                     dtype=tf.float64)
            inter1 = tf.square(tf.matmul(self.X, self.v))
            inter2 = tf.matmul(tf.square(self.X), tf.square(self.v))
            self.interaction_item = tf.reduce_sum(tf.subtract(inter1, inter2), axis = 1 , keep_dims=True)/2

        self.y_out = tf.add(self.linear_item, self.interaction_item)
        self.y_out_prob = tf.nn.sigmoid(self.y_out)

    def add_loss(self):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_out_prob)
        mean_loss = tf.reduce_mean(cross_entropy)

        # 收集正则化损失
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([mean_loss] + reg_loss, name="loss")
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
        """
        tf.cast(
            x,
            dtype,
            name=None
        )
        将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
        那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
        """
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(self.y_out_prob, 0.2), tf.float64), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float64))
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # 添加学习率的指数衰减
        '''
        global_step经常在滑动平均，学习速率变化的时候需要用到，这个参数在
        tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
        里面有，系统会自动更新这个参数的值，从1开始。
        '''
        self.global_step = tf.Variable(0, trainable=False)
        # optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
        #                                    l2_regularization_strength=self.reg_l2)
        # optimizer = tf.train.AdagradOptimizer(self.lr, initial_accumulator_value=1e-9)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # 获取到更新操作步骤
        '''
       在有些机器学习程序中我们想要指定某些操作执行的依赖关系，这时我们可以使用tf.control_dependencies()来实现。  
       control_dependencies(control_inputs)返回一个控制依赖的上下文管理器，使用with关键字可以让在这个上下文环境中的操作都在control_inputs 执行。   
       with g.control_dependencies([a, b, c]):
         # `d` and `e` will only run after `a`, `b`, and `c` have executed.
         d = ...
       '''
        with tf.control_dependencies(extra_update_ops):
            self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()

def train_model(sess, model):
    # 导入归一化数据
    normal_train_data = pd.read_csv('../../data/normalized_train_data.csv', header=None)
    train_data = normal_train_data.values

    epochs = 200
    for i in range(epochs):
        iter_index = round(len(train_data)/model.batch_size)

        for k in range(iter_index):
            batch_index = np.random.choice(len(train_data), size=model.batch_size, replace=False)
            batch_data = train_data[batch_index, :]

            feed_data = {model.X: batch_data[:, 0:15],
                         model.y: batch_data[:, 15].reshape([-1, 1]),
                         model.keep_prob: 1.0}
            _, loss, accuracy, y, y_out_prob, y_out, w1, v, b, inter = sess.run([model.train_step, model.loss, model.accuracy,
                                                                          model.y, model.y_out_prob, model.y_out,
                                                                          model.w1, model.v, model.b,model.interaction_item],
                                                                          feed_dict=feed_data)
            # iter_auc = roc_auc_score(batch_data[:, 15].reshape([-1, 1]), y_out_prob)
            print('第{}轮下第{}次迭代的平均损失为{},准确率为{}'.format(i, k, loss, accuracy))
                # print(w1, v, b)
                # print(y_out_prob)
                # print(roc_auc_score(train_data[:, 15].reshape([-1, 1]), y_out_prob))
                # print(v)


def test＿model(sess, model):
    #　导入测试数据
    normal＿test＿data = pd.read_csv('../../data/normalized_test_data.csv', header=None)
    test_data = normal＿test＿data.values

    feed_data = {model.X: test_data[:, 0:15],
                 model.y: test_data[:, 15].reshape([-1, 1]),
                 model.keep_prob: 1.0}
    accuracy, y, y_out_prob, y_out, w1, v, b = sess.run([model.accuracy,
                                                                         model.y, model.y_out_prob, model.y_out,
                                                                         model.w1, model.v, model.b],
                                                                        feed_dict=feed_data)
    # iter_auc = roc_auc_score(batch_data[:, 15].reshape([-1, 1]), y_out_prob)
    print('准确率为{}'.format(accuracy))
    y_pred = sess.run(tf.cast(tf.greater_equal(y_out_prob, 0.2), tf.float64))
    print(roc_auc_score(test_data[:, 15].reshape([-1, 1]), y_pred))


if __name__ == '__main__':
    FM_model = FM(
        latent_dims = 20,
        lr = 1e-3,
        batch_size = 5120,
        feature_length = 15,
    )
    FM_model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore trained parameters

        print('开始训练')
        train_model(sess, FM_model)

        print('开始测试')
        test＿model(sess, FM_model)