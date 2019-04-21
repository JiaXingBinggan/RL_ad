# _*_ coding: utf-8 _*_
import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import pandas as pd
from src.config import config

def sigmoid(p):
    return 1.0 / (1.0 + math.exp(-p))


def pred_lr(x):
    p = w_0
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return p


def pred(x):
    p = w_0
    sum_1 = 0
    sum_2 = 0
    for (feat, val) in x:
        tmp = v[feat] * val # val=1
        sum_1 += tmp
        sum_2 += tmp * tmp
    p = np.sum(sum_1 * sum_1 - sum_2) / 2.0 + w_0 # FM基础公式
    for (feat, val) in x:
        p += w[feat] * val
    p = sigmoid(p)
    return (p, sum_1)

# 更新参数w
def update_w(y, p, x, vsum):
    global w_0
    d = y - p
    w_0 = w_0 * (1 - weight_decay) + learning_rate * d
    for (feat, val) in x:
        w[feat] = w[feat] * (1 - weight_decay) + learning_rate * d * val
    for (feat, val) in x:
        v[feat] = v[feat] * (1 - v_weight_decay) + learning_rate * d * (val * vsum - v[feat] * val * val)

# 提取train.txt里的数据，生成元组格式
def one_data_y_x(line):
    s = line.strip().replace(':', ',').split(',')
    y = int(s[config['data_feature_index']*2]) # 如果修改了特征则需要修改这里
    x = []
    for i in range(0, len(s)-3, 2): # 后三位分别是click，payprice，hour
        val = 1
        if not one_value:
            val = float(s[i + 1])
        x.append((int(s[i]), val))
    return (y, x)

# 输出模型
def output_model(model_file):
    print('output model to ' + model_file)
    foo = open(model_file, 'w')
    foo.write('%.5f %d %d\n' % (w_0, feature_num, k))
    for i in range(feature_num):
        foo.write('%d %.5f' % (i, w[i]))
        for j in range(k):
            foo.write(' %.5f' % v[i][j])
        foo.write(' %s\n' % index_feature[str(i)])
    foo.close()

    embedding_df = pd.DataFrame(data=v)
    embedding_df.to_csv('../../data/fm/embedding_v.csv', header=None, index=None)


def load_model(model_file):
    global feature_num, k, w_0, w, v, index_feature, feature_index
    print('loading model from ' + model_file)
    fi = open(model_file, 'r')
    line_num = 0
    for line in fi:
        line_num += 1
        s = line.strip().split()
        if line_num == 1:
            w_0 = float(s[0])
            feature_num = int(s[1])
            k = int(s[2])
            v = np.zeros((feature_num, k))
            w = np.zeros(feature_num)
            index_feature = {}
            feature_index = {}
        else:
            i = int(s[0])
            w[i] = float(s[1])
            for j in range(2, 2 + k):
                v[i][j] = float(s[j])
            feature = s[2 + k]
            index_feature[i] = feature
            feature_index[feature] = i
    fi.close()


# 将预测结果写入文件
def pred_to_sub(y_pred, filename):
    with open('../../data/fm/' + filename, 'w') as fo:
        fo.write('id,prob\n')
        for t, prob in enumerate(y_pred, start=1):
            fo.write('{0},{1}\n'.format(t, prob))


# start here

data_path = '../../data/fm'
out_path = '../../data/fm'

f1 = '{0}/train.txt'.format(data_path)
f2 = '{0}/test.txt'.format(data_path)
f3 = '{0}/featindex.txt'.format(data_path)
f4 = '{0}/model_file.txt'.format(data_path)

# global setting
np.random.seed(10)
one_value = True
k = 10  # 隐含因子个数
learning_rate = 0.01  # 学习率
weight_decay = 1E-6
v_weight_decay = 1E-6
train_rounds = 30  # 训练轮数
buffer_num = 100000

# initialise
feature_index = {}
index_feature = {}
max_feature_index = 0


def get_cnt(f): # collections counter
    features = set()
    fi = open(f, 'r')
    for line in fi:
        s = line.replace('\n', '').split('\t')
        features.add(s[1])
        index = s[1]
        index_feature[index] = s[0]
    fi.close()
    feature_cnt = len(features)
    return feature_cnt


# 读取特征个数，用于初始化
feature_num = get_cnt(f3)
print('feature number: ' + str(feature_num))

print('initialising')
init_weight = 0.05
v = (np.random.rand(feature_num, k) - 0.5) * init_weight
w = np.zeros(feature_num)
w_0 = 0

# train
best_auc = 0.
overfitting = False
print('training:')
print('round\tauc\t\tlogloss\t\ttime')
for round in range(1, train_rounds + 1):
    start_time = time.time()
    fi = open(f1, 'r')
    line_num = 0
    train_data = []
    print('第{}轮'.format(round))
    while True:
        '''
        Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        '''
        '''
        1.read() 用法：从文件当前位置起读取size个字节，若无参数size，则表示读取至文件结束为止，它范围为字符串对象。
        2.readline()用法：该方法每次读出一行内容，所以，读取时占用内存小，比较适合大文件，该方法返回一个字符串对象。
        3.readlines()用法：读取整个文件所有行，保存在一个列表(list)变量中，每行作为一个元素，但读取大文件会比较占内存。
        '''
        line = fi.readline().strip()
        if len(line) > 0:
            line_num = (line_num + 1) % buffer_num
            train_data.append(one_data_y_x(line))
        if line_num == 0 or len(line) == 0:
            for data in train_data:
                y = data[0]
                x = data[1]
                # train one data
                (p, vsum) = pred(x)
                update_w(y, p, x, vsum)  # 更新权值
            train_data = []
        if len(line) == 0:
            break
    fi.close()
    train_time = time.time() - start_time
    train_min = int(train_time / 60)
    train_sec = int(train_time % 60)

    # test for this round
    y_true = []
    y_pred = []
    fi = open(f2, 'r')
    for line in fi:
        data = one_data_y_x(line)
        clk = data[0]
        pclk = pred(data[1])[0]
        y_true.append(clk)
        y_pred.append(pclk)
    fi.close()
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    print('%d\t%.8f\t%.8f\t%dm%ds' % (round, auc, logloss, train_min, train_sec))
    pred_to_sub(y_pred, 'test_ctr_pred.csv')

    # 计算train数据集的ctr
    y_true = []
    y_pred = []
    fi = open(f1, 'r')
    for line in fi:
        data = one_data_y_x(line)
        clk = data[0]
        pclk = pred(data[1])[0]
        y_true.append(clk)
        y_pred.append(pclk)
    fi.close()
    pred_to_sub(y_pred, 'train_ctr_pred.csv')

    if overfitting and auc < best_auc:
        output_model(f4)
        # pred_to_sub(y_pred)
        break  # stop training when overfitting two rounds already
    if auc > best_auc:
        best_auc = auc
        overfitting = False
    else:
        overfitting = True

print('v:', v)
print('w:', w)
print('w_0', w_0)

# 转换fm编码至embedding编码
print('data loading\n')
train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
train_data.iloc[:, 17] = train_data.iloc[:, 17].astype(int) # 将时间序列设置为Int类型
embedding_v = pd.read_csv("../../data/fm/embedding_v.csv", header=None)
train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop(0, axis=0) # 读取训练数据集中每条数据的pctr
train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float) # ctr为float类型
train_ctr = train_ctr.iloc[:, 1].values
train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs_1.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率

feat_data = []
current_feat_item = [0 for i in range(155)]
for i, item in enumerate(train_data.values):
    current_feat_item[0] = train_ctr[i] * 100
    for k, feat_next in enumerate(item[0: 15]):
        up_k = k * 10
        current_feat_item[1 + up_k: 11 + up_k] = embedding_v.iloc[feat_next, :].values.tolist()
    current_feat_item[151] = item[15]
    current_feat_item[152] = item[16]
    current_feat_item[153] = item[17]
    current_feat_item[154] = train_ctr[i]
    feat_data.append(current_feat_item)
    current_feat_item = [0 for i in range(155)]

feat_data_df = pd.DataFrame(data=feat_data)
feat_data_df.to_csv('../../data/fm/train_fm_embedding.csv', header=None, index=None)

print('data loading\n')
test_data = pd.read_csv("../../data/fm/test_fm.csv", header=None)
test_data.iloc[:, 17] = test_data.iloc[:, 17].astype(int) # 将时间序列设置为Int类型
embedding_v = pd.read_csv("../../data/fm/embedding_v.csv", header=None)
test_ctr = pd.read_csv("../../data/fm/test_ctr_pred.csv", header=None).drop(0, axis=0) # 读取训练数据集中每条数据的pctr
test_ctr.iloc[:, 1] = test_ctr.iloc[:, 1].astype(float) # ctr为float类型
test_ctr = test_ctr.iloc[:, 1].values
test_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs_1.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率

feat_data = []
current_feat_item = [0 for i in range(155)]
for i, item in enumerate(test_data.values):
    current_feat_item[0] = test_ctr[i] * 100
    for k, feat_next in enumerate(item[0: 15]):
        up_k = k * 10
        current_feat_item[1 + up_k: 11 + up_k] = embedding_v.iloc[feat_next, :].values.tolist()
    current_feat_item[151] = item[15]
    current_feat_item[152] = item[16]
    current_feat_item[153] = item[17]
    current_feat_item[154] = test_ctr[i]
    feat_data.append(current_feat_item)
    current_feat_item = [0 for i in range(155)]

feat_data_df = pd.DataFrame(data=feat_data)
feat_data_df.to_csv('../../data/fm/test_fm_embedding.csv', header=None, index=None)
