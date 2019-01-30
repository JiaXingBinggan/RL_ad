# _*_ coding:utf-8 _*_

import csv
import pandas as pd

# 训练数据27个特征
with open('../../data/train.csv', 'w', newline='') as csvfile: # newline防止每两行就空一行
    spamwriter = csv.writer(csvfile, dialect='excel') # 读要转换的txt文件，文件每行各词间以@@@字符分隔
    with open('../../1458/train.log.txt', 'r') as filein:
        # line1 = [i for i in range(0, 29)]
        # spamwriter.writerow(line1)
        for i, line in enumerate(filein):
            # if i == 0:
            #     continue
            line_list = line.strip('\n').split('\t')
            # line_list[line_list.index('null')] = 'other' #获取下标
            spamwriter.writerow(line_list)
print('train-data读写完毕')

# 测试数据29个特征，多了nclick,nconversation
with open('../../data/test.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, dialect='excel') # 读要转换的txt文件，文件每行各词间以@@@字符分隔
    with open('../../1458/test.log.txt', 'r') as filein:
        for i, line in enumerate(filein):
            # if i == 0:
            #     continue
            line_list = line.strip('\n').split('\t')
            spamwriter.writerow(line_list)
print('test-data读写完毕')

train_header = ['click', 'weekday', 'hour', 'bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent', 'IP',
                'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth', 'slotheight',
                'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice', 'payprice', 'keypage',
                'advertiser', 'usertag']
test_header = ['click', 'weekday', 'hour', 'bidid', 'timestamp', 'logtype', 'ipinyouid', 'useragent',
               'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid', 'slotid', 'slotwidth',
               'slotheight', 'slotvisibility', 'slotformat', 'slotprice', 'creative', 'bidprice',
               'payprice', 'keypage', 'advertiser', 'usertag', 'nclick', 'nconversation']
'''
0606 星期4
0607 星期5
0608 星期6
0609 星期0
0610 星期1
0611 星期2
0612 星期3
'''
# 选择一天的数据作为训练集和测试集
train_data = pd.read_csv('../../data/train.csv')
one_day_data = train_data[train_data.iloc[:, 1].isin([4])] # 选择特定值所在的行
one_day_data.to_csv('../../data/20130606_train_data.csv', index=None)
print(len(one_day_data), one_day_data.iloc[:, 0].sum(), one_day_data.iloc[:, 23].sum())

# 从训练集中选一天作为测试集
test_data = pd.read_csv('../../data/train.csv')
one_day_test_data = test_data[test_data.iloc[:, 1].isin([5])] # 选择特定值所在的行
one_day_test_data.to_csv('../../data/20130607_test_data.csv', index=None)
print(len(one_day_test_data), one_day_test_data.iloc[:, 0].sum(), one_day_test_data.iloc[:, 23].sum())

# test_data = pd.read_csv('../../data/test.csv')
# one_day_test_data = test_data[test_data.iloc[:, 1].isin([4])] # 选择特定值所在的行
# one_day_test_data.to_csv('../../data/20130613_test_data.csv', index=None)

train_data = pd.read_csv('../../data/20130611_test_data.csv', header=None).drop(0, axis=0)
train_data.iloc[:, [0, 23]] = train_data.iloc[:, [0, 23]].astype(int) # 类型强制转换
print(len(train_data), train_data.iloc[:, 0].sum(), train_data.iloc[:, 23].sum())
test_data = pd.read_csv('../../data/20130612_test_data.csv', header=None).drop(0, axis=0)
test_data.iloc[:, [0, 23]] = test_data.iloc[:, [0, 23]].astype(int)
print(len(test_data), test_data.iloc[:, 0].sum(), test_data.iloc[:, 23].sum())

train_data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None).drop(0, axis=0)
train_data.iloc[:, [0, 23]] = train_data.iloc[:, [0, 23]].astype(int) # 类型强制转换
print(len(train_data), train_data.iloc[:, 0].sum(), train_data.iloc[:, 23].sum())
test_data = pd.read_csv('../../sample/20130612_test_sample.csv', header=None).drop(0, axis=0)
test_data.iloc[:, [0, 23]] = test_data.iloc[:, [0, 23]].astype(int)
print(len(test_data), test_data.iloc[:, 0].sum(), test_data.iloc[:, 23].sum())
