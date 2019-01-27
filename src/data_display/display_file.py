import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train = pd.read_csv('../../result/DQN/profits/train_0.015625.txt', header=None)
for i in range(len(train.values)):
    output_str = ''
    for k in range(len(train.values[i, :])):
        output_str += str(train.values[i, k]) + '\t'
    print(output_str)

standard = pd.read_csv('../../result/DQN/profits/result_0.015625.txt', header=None)
for i in range(len(standard.values)):
    output_str = ''
    for k in range(len(standard.values[i, :])):
        output_str += str(standard.values[i, k]) + '\t'
    print(output_str)

standard_hour = pd.read_csv('../../result/DQN/profits/test_hour_clks_0.015625.csv', header=None)
for i in range(len(standard_hour.values)):
    output_str = ''
    for k in range(len(standard_hour.values[i, :])):
        output_str += str(standard_hour.values[i, k]) + '\t'
    print(output_str)

standard_hour = pd.read_csv('../../result/DQN/profits/train_hour_clks_0.015625.csv', header=None)
for i in range(len(standard_hour.values)):
    output_str = ''
    for k in range(len(standard_hour.values[i, :])):
        output_str += str(standard_hour.values[i, k]) + '\t'
    print(output_str)

train_action_ctr = pd.read_csv('../../result/DQN/profits/train_ctr_action_0.015625.csv', header=None)
for i in range(len(train_action_ctr.values)):
    output_str = ''
    for k in range(len(train_action_ctr.values[i, :])):
        output_str += str(train_action_ctr.values[i, k]) + '\t'
    print(output_str)

print('...test\n')
test_action_ctr = pd.read_csv('../../result/DQN/profits/test_ctr_action_0.015625.csv', header=None)
for i in range(len(test_action_ctr.values)):
    output_str = ''
    for k in range(len(test_action_ctr.values[i, :])):
        output_str += str(test_action_ctr.values[i, k]) + '\t'
    print(output_str)

base = pd.read_csv('../../result/results.best.perf.txt', header=None)
for i in range(len(base.values)):
    print(base.values[i][0])

train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
train_data.iloc[:, 17] = train_data.iloc[:, 17].astype(int)
train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率
train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float)
train_ctr = train_ctr.iloc[:, 1].values

for i in range(len(train_data)):
    auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()

    # auction所在小时段索引
    hour_index = auc_data[17]
    current_data_ctr = float(train_ctr[i])  # 当前数据的ctr，原始为str，应该转为float
    if current_data_ctr >= train_avg_ctr[int(hour_index)]: # 才获取数据state
        print(i)
        # for k in range(i+1, len(train_data)):
        #     auc_data = train_data.iloc[k: k + 1, :].values.flatten().tolist()
        #     if float(train_ctr[k]) >= train_avg_ctr[int(auc_data[17])]:
        #         print('下一个')
        #         print(k)
        #         break
        auc_datas = train_data.iloc[i+1:, :].values
        # print(auc_datas[:, 17])
        compare_ctr = train_ctr[i+1:] >= train_avg_ctr[auc_datas[:, 17]]
        print(len(train_avg_ctr[auc_datas[:, 17]]))
        # print(np.where(compare_ctr == True))
        # print(np.where(compare_ctr == True)[0][0] + (i+1))
        # print('\n')
train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
train_data.iloc[:, 17] = train_data.iloc[:, 17].astype(int) # 将时间序列设置为Int类型
train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop(0, axis=0) # 读取训练数据集中每条数据的pctr
train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float) # ctr为float类型
train_ctr = train_ctr.iloc[:, 1].values
train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率

hour_index = train_data.iloc[:, 17]
print(np.sum(train_ctr > train_avg_ctr[hour_index])) # ctr大于平均ctr的数量

test_data = pd.read_csv("../../data/fm/test_fm.csv", header=None)
test_data.iloc[:, 17] = test_data.iloc[:, 17].astype(int) # 将时间序列设置为Int类型
test_ctr = pd.read_csv("../../data/fm/test_ctr_pred.csv", header=None).drop(0, axis=0) # 读取训练数据集中每条数据的pctr
test_ctr.iloc[:, 1] = test_ctr.iloc[:, 1].astype(float) # ctr为float类型
test_ctr = test_ctr.iloc[:, 1].values
test_avg_ctr = pd.read_csv("../../transform_precess/test_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率

hour_index = test_data.iloc[:, 17]
print(np.sum(test_ctr > test_avg_ctr[hour_index]))

train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
train_data.iloc[:, 17] = train_data.iloc[:, 17].astype(int)
train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率
train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float)
train_ctr = train_ctr.iloc[:, 1].values

compare_budget = 0
for i in range(len(train_data)):
    auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()

    # auction所在小时段索引
    hour_index = auc_data[17]
    current_data_ctr = float(train_ctr[i])  # 当前数据的ctr，原始为str，应该转为float
    if current_data_ctr >= train_avg_ctr[int(hour_index)]: # 才获取数据state
        compare_budget += auc_data[17]
print(compare_budget)

test_data = pd.read_csv("../../data/fm/test_fm.csv", header=None)
test_data.iloc[:, 17] = test_data.iloc[:, 17].astype(int)
test_ctr = pd.read_csv("../../data/fm/test_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
test_avg_ctr = pd.read_csv("../../transform_precess/test_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率
test_ctr.iloc[:, 1] = test_ctr.iloc[:, 1].astype(float)
test_ctr = test_ctr.iloc[:, 1].values

compare_budget = 0
for i in range(len(test_data)):
    auc_data = test_data.iloc[i: i + 1, :].values.flatten().tolist()

    # auction所在小时段索引
    hour_index = auc_data[17]
    current_data_ctr = float(test_ctr[i])  # 当前数据的ctr，原始为str，应该转为float
    if current_data_ctr >= test_avg_ctr[int(hour_index)]: # 才获取数据state
        compare_budget += auc_data[17]
print(compare_budget)