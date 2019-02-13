import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import config

# 统计数据
train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
train_data.iloc[:, config['data_hour_index']] = train_data.iloc[:, config['data_hour_index']].astype(int)
train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率
train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float)
train_ctr =  train_ctr.reset_index(drop=True)
train_ctr = train_ctr.iloc[:, 1].values

print(len(train_data), np.sum(train_data.iloc[:, config['data_clk_index']]), np.sum(train_data.iloc[:, config['data_marketprice_index']]))
hour_index = train_data.iloc[:, config['data_hour_index']]
print('训练集ctr大于平均ctr的数量', np.sum(train_ctr >  train_avg_ctr[hour_index]))
print('训练集ctr大于平均ctr的曝光花费', np.sum(train_data[train_ctr > train_avg_ctr[hour_index]].iloc[:, config['data_marketprice_index']]))
with_clk_index = train_data.iloc[:, config['data_clk_index']].isin([1])
with_clk_hour_index = train_data[with_clk_index].iloc[:, config['data_hour_index']]
print('训练集ctr大于点击/曝光ctr的数量', np.sum(train_ctr[with_clk_index.values] > train_avg_ctr[with_clk_hour_index]))
train_ctr_mprice_data = {'ctr': train_ctr[with_clk_index.values],
                        'marketprice': train_data[with_clk_index].iloc[:, config['data_marketprice_index']].values,
                        'hour_index': train_data[with_clk_index].iloc[:, config['data_hour_index']].values}
train_ctr_mprice_data_df = pd.DataFrame(data=train_ctr_mprice_data)
train_ctr_mprice_data_df.to_csv('../../data/data_statics/train_ctr_mprice_data.csv', index=None)

# 统计各时段ctr及出价平均值，以及点击数量
clk_nums = []
bid_avgs = []
ctr_avgs = []
hour_arrays = []
for i in range(0, 24):
    data_stastics = np.sum(train_ctr_mprice_data_df[train_ctr_mprice_data_df.iloc[: ,1].isin([i])])
    clk_num = len(train_ctr_mprice_data_df[train_ctr_mprice_data_df.iloc[: ,1].isin([i])])
    ctr_avg = data_stastics['ctr'] / clk_num
    bid_avg = data_stastics['marketprice'] / clk_num
    clk_nums.append(clk_num)
    ctr_avgs.append(ctr_avg)
    bid_avgs.append(bid_avg)
    hour_arrays.append(i)
hour_data_statics = {'clk_nums': clk_nums, 'ctr_avgs': ctr_avgs, 'bid_avgs': bid_avgs, 'hour_arrays': hour_arrays}
hour_data_statics_df = pd.DataFrame(data=hour_data_statics)
hour_data_statics_df.to_csv('../../data/data_statics/train_hour_data_statics.csv', index=None)
print('训练集平均有点击出价均值', np.sum(train_data[train_data.iloc[:, config['data_clk_index']].isin([1])].iloc[:, config['data_marketprice_index']])/328) # 平均有点击出价均值为90.51219512195122

test_data = pd.read_csv("../../data/fm/test_fm.csv", header=None)
test_data.iloc[:, config['data_hour_index']] = test_data.iloc[:, config['data_hour_index']].astype(int)
test_ctr = pd.read_csv("../../data/fm/test_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
test_avg_ctr = pd.read_csv("../../transform_precess/test_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率
test_ctr.iloc[:, 1] = test_ctr.iloc[:, 1].astype(float)
test_ctr =  test_ctr.reset_index(drop=True)
test_ctr = test_ctr.iloc[:, 1].values

print(len(test_data), np.sum(test_data.iloc[:, config['data_clk_index']]), np.sum(test_data.iloc[:, config['data_marketprice_index']]))
hour_index = test_data.iloc[:, config['data_hour_index']]
print('测试集ctr大于训练数据平均ctr的曝光花费', np.sum(test_data[test_ctr >= train_avg_ctr[hour_index]].iloc[:, config['data_marketprice_index']]))
print('测试集ctr大于训练数据平均ctr的数量', np.sum(test_ctr >=  train_avg_ctr[hour_index]))

with_clk_index = test_data.iloc[:, config['data_clk_index']].isin([1])
test_ctr_mprice_data = {'ctr': test_ctr[with_clk_index.values],
                        'marketprice': test_data[with_clk_index].iloc[:, config['data_marketprice_index']].values,
                        'hour_index': test_data[with_clk_index].iloc[:, config['data_hour_index']].values}
test_ctr_mprice_data_df = pd.DataFrame(data=test_ctr_mprice_data)
test_ctr_mprice_data_df.to_csv('../../data/data_statics/test_ctr_mprice_data.csv', index=None)

# 统计各时段ctr及出价平均值，以及点击数量
clk_nums = []
bid_avgs = []
ctr_avgs = []
hour_arrays = []
for i in range(0, 24):
    data_stastics = np.sum(test_ctr_mprice_data_df[test_ctr_mprice_data_df.iloc[: ,1].isin([i])])
    clk_num = len(test_ctr_mprice_data_df[test_ctr_mprice_data_df.iloc[: ,1].isin([i])])
    ctr_avg = (data_stastics['ctr'] / clk_num) if clk_num != 0 else 0
    bid_avg = (data_stastics['marketprice'] / clk_num) if clk_num != 0 else 0
    clk_nums.append(clk_num)
    ctr_avgs.append(ctr_avg)
    bid_avgs.append(bid_avg)
    hour_arrays.append(i)
test_hour_data_statics = {'clk_nums': clk_nums, 'ctr_avgs': ctr_avgs, 'bid_avgs': bid_avgs, 'hour_arrays': hour_arrays}
test_hour_data_statics_df = pd.DataFrame(data=test_hour_data_statics)
hour_index = test_data[with_clk_index].iloc[:, config['data_hour_index']]

print('测试集具有点击的数据大于各时段平均ctr（为各数据ctr的平均值）的数量', np.sum(test_ctr[with_clk_index.values] >= test_hour_data_statics_df.iloc[hour_index, 2]))
print('测试集具有点击的数据大于0.0005的数量', np.sum(test_ctr[with_clk_index.values] >= 0.0005))
print('测试集具有点击的数据大于训练数据时段点击/曝光ctr的数量', np.sum(test_ctr[with_clk_index.values] >= train_avg_ctr[hour_index]))
test_hour_data_statics_df.to_csv('../../data/data_statics/test_hour_data_statics.csv', index=None)
print('测试集平均有点击出价均值', np.sum(test_data[test_data.iloc[:, config['data_clk_index']].isin([1])].iloc[:, config['data_marketprice_index']])/307) # 均有点击出价均值为89.34201954397395

x_axis = np.arange(0,24)
train_y_aixs_1 = hour_data_statics_df.iloc[:, 1].values
train_y_aixs_2 = train_avg_ctr

plt.plot(x_axis, train_y_aixs_1, 'r', label='train average ctr(one hour clk / imps)')
plt.plot(x_axis, train_y_aixs_2, 'b', label='train data average ctr(one hour sum_ctr / imps)')
plt.legend()
plt.show()

test_y_aixs_1 = test_hour_data_statics_df.iloc[:, 1].values
test_y_aixs_2 = test_avg_ctr

plt.plot(x_axis, test_y_aixs_1, 'r', label='train average ctr(one hour clk / imps)')
plt.plot(x_axis, test_y_aixs_2, 'b', label='train data average ctr(one hour sum_ctr / imps)')
plt.legend()
plt.show()

# 查看各时段平均ctr与数据集中ctr的关系
# x_axis = np.arange(0, 328)
# train_y_axis_1 = train_ctr_mprice_data_df.iloc[:, 0].values
# train_y_axis_2 = []
# hour_data_statics_df.iloc[:, 0] = hour_data_statics_df.iloc[:, 0].astype(int)
# for i, clk_nums in enumerate(hour_data_statics_df.iloc[:, 0].values):
#     for clk_num in range(0, clk_nums):
#         train_y_axis_2.append(test_avg_ctr[i])
# plt.plot(x_axis, train_y_axis_1, 'rx', label='real data average ctr(one hour sum_ctr / imps)')
# plt.plot(x_axis, train_y_axis_2, 'b', label='=average ctr(one hour clk / imps)')
# plt.legend()
# plt.show()

# 查看数据中ctr与出价的对比
x_axis = train_ctr_mprice_data_df.iloc[:, 0].values
y_axis = train_ctr_mprice_data_df.iloc[:, 1].values
plt.plot(x_axis, y_axis, 'bx', label='ctr - price')
plt.legend()
plt.show()