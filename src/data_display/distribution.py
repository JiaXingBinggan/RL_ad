import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import config

# 统计数据
train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
train_data.iloc[:, config['data_hour_index']] = train_data.iloc[:, config['data_hour_index']].astype(int)
train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float)
train_ctr =  train_ctr.reset_index(drop=True)
# train_ctr = train_ctr.iloc[:, 1].values
train_hour_index = train_data.iloc[:, config['data_hour_index']]

with_clk_index = train_data.iloc[:, config['data_clk_index']].isin([1])

test_data = pd.read_csv("../../data/fm/test_fm.csv", header=None)
test_data.iloc[:, config['data_hour_index']] = test_data.iloc[:, config['data_hour_index']].astype(int)
test_ctr = pd.read_csv("../../data/fm/test_ctr_pred.csv", header=None).drop([0], axis=0) # 读取训练数据集中每条数据的pctr
test_ctr.iloc[:, 1] = test_ctr.iloc[:, 1].astype(float)
test_ctr =  test_ctr.reset_index(drop=True)
# test_ctr = test_ctr.iloc[:, 1].values

test_hour_index = test_data.iloc[:, config['data_hour_index']]
test_with_clk_index = test_data.iloc[:, config['data_clk_index']].isin([1])

time_aixs = np.arange(0, 24)

# 各时段曝光数Imps distribution
train_imps_time_rates = [] # 各时段曝光数占比
test_imps_time_rates = [] # 各时段曝光数占比
for i in time_aixs:
    is_train_time_index = train_data.iloc[:, config['data_hour_index']].isin([i])
    is_test_time_index = test_data.iloc[:, config['data_hour_index']].isin([i])

    train_imps_time_rate = len(train_data[is_train_time_index].values)/len(train_data)
    train_imps_time_rates.append(train_imps_time_rate)
    test_imps_time_rate = len(test_data[is_test_time_index].values) / len(test_data)
    test_imps_time_rates.append(test_imps_time_rate)
plt.plot(time_aixs, train_imps_time_rates, 'r', label='train imps_distribution')
plt.plot(time_aixs, test_imps_time_rates, 'b', label='test imps_distribution')
plt.legend()
plt.savefig('./plot_images/imps_distribution.png')
plt.show()

# 各时段平均成交价avg_market_price distribution
train_time_avg_market_prices = []
test_time_avg_market_prices = []
for i in time_aixs:
    is_train_time_index = train_data.iloc[:, config['data_hour_index']].isin([i])
    is_test_time_index = test_data.iloc[:, config['data_hour_index']].isin([i])

    train_time_avg_market_price = np.sum(train_data[is_train_time_index].iloc[:, config['data_marketprice_index']])/len(
        train_data[is_train_time_index])
    train_time_avg_market_prices.append(train_time_avg_market_price)
    test_time_avg_market_price = np.sum(test_data[is_test_time_index].iloc[:, config['data_marketprice_index']]) / len(
        test_data[is_test_time_index])
    test_time_avg_market_prices.append(test_time_avg_market_price)
plt.plot(time_aixs, train_time_avg_market_prices, 'r', label='train avg_market_price_distribution')
plt.plot(time_aixs, test_time_avg_market_prices, 'b', label='test avg_market_price_distribution')
plt.legend()
plt.savefig('./plot_images/avg_market_price_distribution.png')
plt.show()

# 各时段平均avg_ctr distribution
train_time_avg_ctrs = []
test_time_avg_ctrs = []
for i in time_aixs:
    is_train_time_index = train_data.iloc[:, config['data_hour_index']].isin([i])
    is_test_time_index = test_data.iloc[:, config['data_hour_index']].isin([i])

    train_time_avg_ctr = np.sum(train_ctr[is_train_time_index].iloc[:, 1]) / len(train_ctr[is_train_time_index])
    test_time_avg_ctr = np.sum(test_ctr[is_test_time_index].iloc[:, 1]) / len(test_ctr[is_test_time_index])
    train_time_avg_ctrs.append(train_time_avg_ctr)
    test_time_avg_ctrs.append(test_time_avg_ctr)
plt.plot(time_aixs, train_time_avg_ctrs, 'r', label='train avg_ctr_distribution')
plt.plot(time_aixs, test_time_avg_ctrs, 'b', label='test avg_ctr_distribution')
plt.legend()
plt.savefig('./plot_images/avg_ctr_distribution.png')
plt.show()

# 各时段clk_rate distribution
train_time_clk_rates = []
test_time_clk_rates = []
for i in time_aixs:
    is_train_time_index = train_data.iloc[:, config['data_hour_index']].isin([i])
    is_test_time_index = test_data.iloc[:, config['data_hour_index']].isin([i])
    train_time_clk_rate = np.sum(train_data[is_train_time_index].iloc[:, config['data_clk_index']]) / np.sum(train_data.iloc[:, config['data_clk_index']])
    test_time_clk_rate = np.sum(test_data[is_test_time_index].iloc[:, config['data_clk_index']]) / np.sum(test_data.iloc[:, config['data_clk_index']])
    train_time_clk_rates.append(train_time_clk_rate)
    test_time_clk_rates.append(test_time_clk_rate)
plt.plot(time_aixs, train_time_clk_rates, 'r', label='train clk_rate_distribution')
plt.plot(time_aixs, test_time_clk_rates, 'b', label='test clk_rate_distribution')
plt.legend()
plt.savefig('./plot_images/ctr_rate_distribution.png')
plt.show()

# marketprice_num distribution
train_ctr_num = []
for i in np.arange(1, 301, 5):
    train_ctr_num.append(np.sum(train_data.iloc[:, config['data_marketprice_index']].values > i)/len(train_data))
test_ctr_num = []
for i in np.arange(1, 301, 5):
    test_ctr_num.append(np.sum(test_data.iloc[:, config['data_marketprice_index']].values > i)/len(test_data))
x_axis = np.arange(1, 301, 5)
plt.plot(x_axis, train_ctr_num, 'r', label='train marketprice_distribution')
plt.plot(x_axis, test_ctr_num, 'b', label='test marketprice_distribution')
plt.legend()
plt.savefig('./plot_images/marketprice_distribution.png')
plt.show()

