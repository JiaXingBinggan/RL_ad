'''
定义一些超参数
'''
import pandas as pd
import numpy as np

config = {
    'e_greedy': 1,
    'learning_rate': 0.1,
    'pg_learning_rate': 1e-3,
    'reward_decay': 1,
    'feature_num': 153,
    'data_pctr_index': 154,
    'data_hour_index': 153, # 17:train-fm
    'data_clk_index': 151, # 15:train-fm
    'data_marketprice_index': 152, # 16:train-fm
    'data_feature_index': 151, # 15:train-fm
    'state_feature_num': 151,
    'train_date': str(20130606), # sample 328481 328 22067108
    'test_date': str(20130607), # sample 307176 307 19441889
    'train_budget': 22067108, # 22067108
    'train_auc_num': 127594, # 155444, 127594, 173710
    'test_budget': 22067108, # 14560732
    'test_auc_num': 328481, # 68244
    'budget_para': [1/2],
    'train_episodes': 300,
    'neuron_nums': 100,
    'GPU_fraction': 1,
    'relace_target_iter': 1000,
    'memory_size': 500000,
    'batch_size': 32, # GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128...时往往要比设置为整10、整100的倍数时表现更优
}
# train_data = pd.read_csv('../sample/20130607_test_data.csv', header=None).drop([0])
# # # price_counter_train = []
# # # for i in range(0, 301):
# # #     sum_price = np.sum(train_data.iloc[:, 23].isin([i]))
# # #     price_counter_train.append(sum_price)
# # #     print(len(price_counter_train))
# # # print(price_counter_train)
# #
# train_ctr = pd.read_csv('../data/fm/test_ctr_pred.csv', header=None).drop([0])
#
# data = {'clk':train_data.iloc[:, 0].values, 'price': train_data.iloc[:, 23].values, 'ctr': train_ctr.iloc[:, 1].values}
#
# dataframe = pd.DataFrame(data=data)
# dataframe.to_csv('test.csv', header=None, index=None)
#
# to_data = []
# f_in = open('test1.csv', 'w')
# for data in open('test.csv'):
#     f_in.write(data.replace(',', ' '))
#
# train_data = pd.read_csv('../sample/20130606_train_sample.csv', header=None).drop([0])
# # price_counter_train = []
# # for i in range(0, 301):
# #     sum_price = np.sum(train_data.iloc[:, 23].isin([i]))
# #     price_counter_train.append(sum_price)
# #     print(len(price_counter_train))
# # print(price_counter_train)
#
# train_ctr = pd.read_csv('../data/fm/train_ctr_pred.csv', header=None).drop([0])
#
# data = {'clk':train_data.iloc[:, 0].values, 'price': train_data.iloc[:, 23].values, 'ctr': train_ctr.iloc[:, 1].values}
#
# dataframe = pd.DataFrame(data=data)
# dataframe.to_csv('test22.csv', header=None, index=None)
#
# to_data = []
# f_in = open('test2.csv', 'w')
# for data in open('test22.csv'):
#     f_in.write(data.replace(',', ' '))