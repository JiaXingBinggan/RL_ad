import pandas as pd
import numpy as np
from src.config import config

# 统计每天的每小时的点击数，所有有点击的数据的支付价以及ctr
train_data = pd.read_csv('../../sample/'+config['train_date']+'_train_sample.csv', header=None).drop(0, axis=0)
train_data.iloc[:, [0, 2]] = train_data.iloc[:, [0, 2]].astype(int) # 类型强制转换
train_ctr = pd.read_csv('../../data/fm/train_ctr_pred.csv', header=None).drop(0, axis=0)

hour_clks = []
for i in range(24):
    hour_data = train_data[train_data.iloc[:, 2].isin([i])]
    clk_is_one_index = hour_data.iloc[:, 0].isin([1])
    clks_data = hour_data[clk_is_one_index]
    hour_clks.append(len(clks_data))
out_str = ''
for hour_clk in hour_clks:
    out_str += (str(hour_clk) + '\t')
print(out_str)
clk_is_one_index = train_data.iloc[:, 0].isin([0, 1])
price_data = train_data[clk_is_one_index].iloc[:, 23]
ctr_data = train_ctr[clk_is_one_index].iloc[:, 1]
ctr_price_data = {'ctr': ctr_data, 'price': price_data}
ctr_price_df = pd.DataFrame(data=ctr_price_data)
ctr_price_df.to_csv('../../transform_precess/'+config['train_date']+'_train_ctr_clk.csv')

# 测试集
test_data = pd.read_csv('../../sample/'+config['test_date']+'_test_sample.csv', header=None).drop(0, axis=0)
test_data.iloc[:, [0, 2]] = test_data.iloc[:, [0, 2]].astype(int) # 类型强制转换
test_ctr = pd.read_csv('../../data/fm/test_ctr_pred.csv', header=None).drop(0, axis=0)

test_hour_clks = []
for i in range(24):
    hour_data = test_data[test_data.iloc[:, 2].isin([i])]
    clk_is_one_index = hour_data.iloc[:, 0].isin([1])
    clks_data = hour_data[clk_is_one_index]
    test_hour_clks.append(len(clks_data))
out_str = ''
for hour_clk in test_hour_clks:
    out_str += (str(hour_clk) + '\t')
print(out_str)
clk_is_one_index = test_data.iloc[:, 0].isin([0, 1])
price_data = test_data[clk_is_one_index].iloc[:, 23]
ctr_data = test_ctr[clk_is_one_index].iloc[:, 1]
ctr_price_data = {'ctr': ctr_data, 'price': price_data}
ctr_price_df = pd.DataFrame(data=ctr_price_data)
ctr_price_df.to_csv('../../transform_precess/'+config['test_date']+'_test_ctr_clk.csv')
