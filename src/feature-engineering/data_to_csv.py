# _*_ coding:utf-8 _*_

import csv
import pandas as pd

# 训练数据27个特征
# with open('../../data/train.csv', 'w', newline='') as csvfile: # newline防止每两行就空一行
#     spamwriter = csv.writer(csvfile, dialect='excel') # 读要转换的txt文件，文件每行各词间以@@@字符分隔
#     with open('../../1458/train.log.txt', 'r') as filein:
#         # line1 = [i for i in range(0, 29)]
#         # spamwriter.writerow(line1)
#         for i, line in enumerate(filein):
#             if i == 0:
#                 continue
#
#             line_list = line.strip('\n').split('\t')
#             # line_list[line_list.index('null')] = 'other' #获取下标
#             spamwriter.writerow(line_list)
# print('train-data读写完毕')
# #
# # 测试数据29个特征，多了nclick,nconversation
# with open('../../data/test.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, dialect='excel') # 读要转换的txt文件，文件每行各词间以@@@字符分隔
#     with open('../../1458/test.log.txt', 'r') as filein:
#         for i, line in enumerate(filein):
#             if i == 0:
#                 continue
#             line_list = line.strip('\n').split('\t')
#             spamwriter.writerow(line_list)
# print('test-data读写完毕')

# 选择一天的数据作为训练集和测试集
train_data = pd.read_csv('../../data/train.csv', header=None)
one_day_data = train_data[train_data.iloc[:, 1].isin([4])] # 选择特定值所在的行
print(len(one_day_data))
# one_day_data.to_csv('../data/20130606_train_data.csv', header=None, index=None)
# print(len(one_day_data), one_day_data.iloc[:, 0].sum(), one_day_data.iloc[:, 23].sum())
#
# test_data = pd.read_csv('../data/test.csv', header=None)
# one_day_test_data = test_data[test_data.iloc[:, 1].isin([4])] # 选择特定值所在的行
# one_day_test_data.to_csv('../data/20130613_test_data.csv', header=None, index=None)
train_data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None)
print(len(train_data), train_data.iloc[:, 0].sum(), train_data.iloc[:, 23].sum())
test_data = pd.read_csv('../../sample/20130613_test_sample.csv', header=None)
print(len(test_data), test_data.iloc[:, 0].sum(), test_data.iloc[:, 23].sum())
