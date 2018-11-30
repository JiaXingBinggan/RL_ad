# 此文件主要用于：根据数据获取每个小时的平均点击率

import pandas as pd

# 训练集下
hour_clks = pd.read_csv('../../transform_precess/hour_select_result.csv',header=None).values
# # 使用python中的pandas求每个值占该列的比例,这是流量占比
# hour_clks.iloc[:, :] = hour_clks.iloc[:, :]/hour_clks.iloc[:, :].sum()
# tansform_p = hour_clks.mean(1) # 各小时转移概率，a.mean(1)表示按行计算平均值
# tansform_p.to_csv('../transform_precess/transform_p.csv', header=None, index=[i for i in range(24)])

avg_ctrs = []
# 每个时间段的平均点击率
date_data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None)
for i in range(24):
    hour_imps = date_data[date_data.iloc[:, 2].isin([i])]
    avg_ctrs.append(hour_clks[0][i]/len(hour_imps))

train_avg_ctr = pd.DataFrame(data=avg_ctrs)
train_avg_ctr.to_csv('../../transform_precess/train_avg_ctrs.csv', header=None, index=None)

# 测试集下
test_hour_clks = pd.read_csv('../../transform_precess/test_hour_select_result.csv',header=None).values

avg_ctrs = []
# 每个时间段的平均点击率
test_date_data = pd.read_csv('../../sample/20130613_test_sample.csv', header=None)
for i in range(24):
    hour_imps = test_date_data[test_date_data.iloc[:, 2].isin([i])]
    avg_ctrs.append(test_hour_clks[0][i]/len(hour_imps))

test_avg_ctr = pd.DataFrame(data=avg_ctrs)
test_avg_ctr.to_csv('../../transform_precess/test_avg_ctrs.csv', header=None, index=None)
