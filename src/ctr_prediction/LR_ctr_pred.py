import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.metrics import log_loss

random.seed(1024)

x = tf.placeholder(name='input', shape=(None, 15), dtype=tf.float64)
ctr_ = tf.placeholder(name='click', shape=(None), dtype=tf.float64)

weight = tf.get_variable(name='weight', shape=[15,1], initializer=tf.truncated_normal_initializer(mean=0, stddev=1), dtype=tf.float64)
bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0), dtype=tf.float64)

ctr_pred = tf.matmul(x, weight) + bias

# 加入正则化项，防止过拟合
regularizer = tf.contrib.layers.l2_regularizer(0.001)
regularization = regularizer(weight)

# 交叉熵
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_, logits=ctr_pred) # 这里的ctr_pred不能加sigmoid函数处理

train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 导入归一化数据
    normal_train_data = pd.read_csv('../../data/normalized_train_data.csv', header=None)

    train_data = normal_train_data.values

    for i in range(500000):
        random_index = random.randint(0, len(train_data))
        feed_data = {x: train_data[:, 0:15],
                    ctr_: train_data[:, 15].reshape([-1, 1])}
        if i % 1000 == 0:
            _, loss_, weight_s, bias_s = sess.run([train_step, cross_entropy, weight, bias], feed_dict=feed_data)
            print('第{}轮的平均损失为{}'.format(i, np.sum(loss_)/len(loss_)))

    print(weight_s, bias_s)

    train_input = train_data[:, 0:15]
    train_ctr_pred = tf.sigmoid(tf.matmul(train_input, weight_s) + bias_s)
    train_ctr_pred_data = pd.DataFrame(sess.run(train_ctr_pred))
    train_ctr_pred_data.to_csv('../../data/train_lr_pred.csv', header=None)

    normal_test_data = pd.read_csv('../../data/normalized_test_data.csv', header=None)
    test_data = normal_test_data.values
    test_input = test_data[:, 0:15]
    test_ctr_pred = tf.sigmoid(tf.matmul(test_input, weight_s) + bias_s)
    test_ctr_pred_data = pd.DataFrame(sess.run(test_ctr_pred))
    test_ctr_pred_data.to_csv('../../data/test_lr_pred.csv', header=None)





