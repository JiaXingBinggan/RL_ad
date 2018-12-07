import tensorflow as tf
import pandas as pd
import numpy as np
import random
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

class FM:
    def __init__(
        self,
        latent_dims, # 隐向量维度
        lr, # 学习率
        batch_size,
        reg_l1, # L1正则化项
        reg_l2, # L2正则化项
        feature_length, # 特征长度
    ):
        self.latent_dims = latent_dims
        self.lr = lr
        self.batch_size = batch_size
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.featuere_length = feature_length

    def add_placeholders(self):
        self.X = tf.placeholder('float32', shape=(None, self.featuere_length))
        self.y = tf.placeholder('int64', shape=(None, 1))
        '''
        为了防止或减轻过拟合,Dropout就是在不同的训练过程中随机扔掉一部分神经元。
        也就是让某个神经元的激活值以一定的概率p，
        让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
        但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
       '''
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        '''
        定义FM框架
        :return: 对每一条数据的预测标签
        '''
        with tf.variable_scope('linaer_layer'):
            b = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.featuere_length, 1], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
            self.linear_item = tf.add(tf.matmul(self.X, w1), b)

        # 定义交叉项
        with tf.variable_scope('interaction_layer'):
            self.v = tf.get_variable('v', shape=[self.featuere_length, self.latent_dims],
                                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.02))
            self.interaction_item = tf.multiply(0.5, tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(self.X, self.v), 2),
                                                                 tf.matmul(tf.pow(self.X, 2), tf.pow(self.v, 2)))))

        self.y_out = tf.add(self.linear_item, self.interaction_item)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
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
        self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # 添加学习率的指数衰减
        '''
        global_step经常在滑动平均，学习速率变化的时候需要用到，这个参数在
        tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
        里面有，系统会自动更新这个参数的值，从1开始。
        '''
        self.global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
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

# 存储已训练完成的参数，如果它们已经存在
# def check_restore_parameters(sess, saver):
#     ckpt = tf.train.get_checkpoint_state('checkponts')
#     if ckpt and ckpt.model_checkpoint_path:
#         logging.info("加载CNN的参数")
#         saver.restore(sess, ckpt.model_checkpoint_path)
#     else:
#         logging.info("加载FM的参数")
#         saver.restore(sess, ckpt.model_checkpoint_path)

def train_model(sess, model, epochs=10, print_every=50):
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)

    # 训练数据
    train_encoder_data = pd.read_csv('../../data/selected_train_data.csv', header=None)
    train_data = train_encoder_data.values

    for e in range(epochs):
        # sample_index = np.random.choice(len(train_data), size=model.batch_size, replace=False)
        # batch_data = train_data[sample_index, :]

        feed_data = {model.X: train_data[:, 0:15], model.y: train_data[:, 15].reshape([-1, 1]), model.keep_prob: 1.0}
        loss, accuracy, summary, global_step, x, y_out, v, _ = sess.run([model.loss, model.accuracy, merged,
                                                            model.global_step, model.X, model.y_out, model.v, model.train_step], feed_dict=feed_data)
        print(train_data[:, 15].reshape([-1, 1]))
        print(sess.run(tf.argmax(train_data[:, 15].reshape([-1, 1]), 1)))

        print(y_out)
        print(sess.run(tf.argmax(y_out, 1)))

        train_writer.add_summary(summary, global_step=global_step)
        print('第{}轮的损失为{}，准确率为{}'.format(e, loss, accuracy))

if __name__ == '__main__':
    FM_model = FM(
        latent_dims = 20,
        lr = 1e-2,
        batch_size = 51200,
        reg_l1 = 2e-2,
        reg_l2 = 2e-2,
        feature_length = 15,
    )
    FM_model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # restore trained parameters

        print('开始训练')
        train_model(sess, FM_model, epochs=200, print_every=500)



