from src.DRLB.env import AD_env
from src.DRLB.RL_brain_for_test import DRLB
from src.DRLB.reward_net_torch import RewardNet
import numpy as np
import pandas as pd
import copy
import datetime
from src.DRLB.config import config

def bid_func(auc_pCTRS, lamda):
    cpc = 30000
    return auc_pCTRS * cpc / lamda


def statistics(B_t, origin_t_spent, origin_t_win_imps,
               origin_t_auctions, origin_t_clks, origin_reward_t, origin_profit_t,  auc_t_datas, bid_arrays, remain_auc_num, t):
    cpc = 30000
    if B_t[t] > 0:
        if B_t[t] - origin_t_spent <= 0 or remain_auc_num[t] - origin_t_auctions <= 0:
            temp_t_auctions = 0
            temp_t_spent = 0
            temp_t_win_imps = 0
            temp_reward_t = 0
            temp_t_clks = 0
            temp_profit_t = 0
            for i in range(len(auc_t_datas)):
                temp_t_auctions += 1
                if remain_auc_num[t] - temp_t_auctions >= 0:
                    if B_t[t] - temp_t_spent >= 0:
                        if auc_t_datas.iloc[i, 2] <= bid_arrays[i]:
                            temp_t_spent += auc_t_datas.iloc[i, 2]
                            temp_t_win_imps += 1
                            temp_t_clks += auc_t_datas.iloc[i, 0]
                            temp_profit_t += (auc_t_datas.iloc[i, 1] * cpc - auc_t_datas.iloc[i, 2])
                            temp_reward_t += auc_t_datas.iloc[i, 0]
                    else:
                        break
                else:
                    break
            t_auctions = temp_t_auctions
            t_spent = temp_t_spent if temp_t_spent > 0 else 0
            t_win_imps = temp_t_win_imps
            t_clks = temp_t_clks
            reward_t = temp_reward_t
            profit_t = temp_profit_t
        else:
            t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t \
                = origin_t_spent, origin_t_win_imps, origin_t_auctions, origin_t_clks, origin_reward_t, origin_profit_t
    else:
        t_auctions = 0
        t_spent = 0
        t_win_imps = 0
        reward_t = 0
        t_clks = 0
        profit_t = 0

    return t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t

def state_(budget, auc_num, auc_t_datas, auc_t_data_pctrs, lamda, B_t, time_t, remain_auc_num):
    cpc = 30000
    bid_arrays = bid_func(auc_t_data_pctrs, lamda)  # 出价
    win_auc_datas = auc_t_datas[auc_t_datas.iloc[:, 2] <= bid_arrays].values  # 赢标的数据
    t_spent = np.sum(win_auc_datas[:, 2])  # 当前t时段花费
    t_auctions = len(auc_t_datas)  # 当前t时段参与拍卖次数
    t_win_imps = len(win_auc_datas)  # 当前t时段赢标曝光数
    t_clks = np.sum(win_auc_datas[:, 0])
    profit_t = np.sum(win_auc_datas[:, 1] * cpc - win_auc_datas[:, 2])  # RewardNet
    reward_t = np.sum(win_auc_datas[:, 0]) # 按点击数作为直接奖励
    # reward_t = np.sum(win_auc_datas[:, 1] * cpc - win_auc_datas[:, 2])  # 按点击数作为直接奖励

    # BCR_t = 0
    if time_t == 0:
        if remain_auc_num[0] > 0:
            if remain_auc_num[0] - t_auctions <= 0:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, 0)
            else:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, 0)
        else:
            t_win_imps = 0
            t_spent = 0
            t_auctions = 0
            reward_t = 0
            t_clks = 0
            profit_t = 0

        B_t[0] = budget - t_spent
        if B_t[0] < 0:
            B_t[0] = 0
        remain_auc_num[0] = auc_num - t_auctions
        if remain_auc_num[0] < 0:
            remain_auc_num[0] = 0
        BCR_t_0 = (B_t[0] - budget) / budget
        BCR_t = BCR_t_0
    else:
        if remain_auc_num[time_t - 1] > 0:
            if remain_auc_num[time_t - 1] - t_auctions <= 0:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, time_t - 1)
            else:
                t_win_imps, t_spent, t_auctions, reward_t, t_clks, profit_t \
                    = statistics(B_t, t_spent, t_win_imps, t_auctions, t_clks, reward_t, profit_t, auc_t_datas, bid_arrays, remain_auc_num, time_t - 1)
        else:
            t_auctions = 0
            t_spent = 0
            t_win_imps = 0
            reward_t = 0
            t_clks = 0
            profit_t = 0

        B_t[time_t] = B_t[time_t - 1] - t_spent
        if B_t[time_t] < 0:
            B_t[time_t] = 0
        remain_auc_num[time_t] = remain_auc_num[time_t - 1] - t_auctions
        if remain_auc_num[time_t] < 0:
            remain_auc_num[time_t] = 0
        BCR_t_current = (B_t[time_t] - B_t[time_t - 1]) / B_t[time_t - 1] if B_t[time_t - 1] > 0 else 0
        BCR_t = BCR_t_current

    ROL_t = 96 - time_t - 1
    CPM_t = t_spent / t_win_imps if t_spent != 0 else 0
    WR_t = t_win_imps / t_auctions if t_auctions > 0 else 0
    state_t = [time_t+1, B_t[time_t], ROL_t, BCR_t, CPM_t, WR_t, reward_t]

    net_reward_t = RewardNet.return_model_reward(state_t)
    # state_t = [(time_t + 1)/96, B_t[time_t]/budget, ROL_t/96, BCR_t, CPM_t/100, WR_t, net_reward_t[0][0]]
    state_t = [time_t + 1, B_t[time_t], ROL_t, BCR_t, CPM_t, WR_t, net_reward_t[0][0]]

    t_real_clks = np.sum(auc_t_datas.iloc[:, 0])

    t_real_imps = len(auc_t_datas)
    return state_t, lamda, B_t, reward_t, profit_t, t_clks, bid_arrays, remain_auc_num, t_win_imps, t_real_imps, t_real_clks, t_spent

def run_test(budget, auc_num, budget_para):
    test_data = pd.read_csv('../../data/DRLB/test_DRLB.csv', header=None).drop([0])
    test_data.iloc[:, [0, 2, 3]] = test_data.iloc[:, [0, 2, 3]].astype(int)
    test_data.iloc[:, [1]] = test_data.iloc[:, [1]].astype(float)

    B_t = [0 for i in range(96)]
    B_t[0] = budget * budget_para

    remain_auc_num = [0 for i in range(96)]
    remain_auc_num[0] = auc_num

    init_lamda = config['init_lamda']
    episode_clks = 0
    episode_real_clks = 0
    episode_imps = 0
    episode_win_imps = 0
    episode_spent = 0
    episode_profit = 0

    temp_lamda_t_next, temp_B_t_next, temp_remain_t_auctions = 0, [], []

    action_records = []

    lamda_record = [init_lamda]

    time_slot_clks = [0 for i in range(96)]
    for t in range(96):
        time_t = t

        # auc_data[0] 是否有点击；auc_data[1] pCTR；auc_data[2] 市场价格； auc_data[3] t划分[1-96]
        auc_t_datas = test_data[test_data.iloc[:, 3].isin([t + 1])]  # t时段的数据
        auc_t_data_pctrs = auc_t_datas.iloc[:, 1].values  # ctrs
        if t == 0:
            state_t, lamda_t, B_t, reward_t, profit_t, t_clks, bid_arrays, t_remain_auc_num, t_win_imps, t_real_imps, t_real_clks, t_spent\
                = state_(budget, auc_num, auc_t_datas, auc_t_data_pctrs,
                                                                         init_lamda, B_t, time_t, remain_auc_num)  # 1时段
            action = RL.choose_best_action(state_t)

            lamda_t_next = lamda_t * (1 + action)

            temp_lamda_t_next, temp_B_t_next, temp_remain_t_auctions = lamda_t_next, B_t, t_remain_auc_num
        else:
            state_t, lamda_t, B_t, reward_t, profit_t, t_clks, bid_arrays, t_remain_auc_num, t_win_imps, t_real_imps, t_real_clks, t_spent\
                = state_(budget, auc_num, auc_t_datas, auc_t_data_pctrs,
                                                                         temp_lamda_t_next, temp_B_t_next, time_t, temp_remain_t_auctions)
            action = RL.choose_best_action(state_t)

            lamda_t_next = lamda_t * (1 + action)


            temp_lamda_t_next, temp_B_t_next, temp_remain_t_auctions = lamda_t_next, B_t, t_remain_auc_num

            if t + 1 == 95:
                lamda_record.append(lamda_t_next)

        action_records.append(action)

        episode_clks += t_clks
        episode_real_clks += t_real_clks
        episode_imps += t_real_imps
        episode_win_imps += t_win_imps
        episode_spent += t_spent
        episode_profit += reward_t
        print('第{}个时段，真实曝光数{}, 赢标数{}, 共获得{}个点击, 真实点击数{}, '
              '利润{}, 预算{}, 花费{}, CPM{}, {}'.format(t + 1, episode_imps, episode_win_imps, episode_clks, episode_real_clks,
                                                   episode_profit, budget, episode_spent,
                                                   episode_spent / episode_win_imps if episode_win_imps > 0 else 0,
                                                   datetime.datetime.now()))

        time_slot_clks[t] = t_clks

    hour_clks = []
    for i in range(96):
        if i == 0:
            continue
        if (i + 1) % 4 == 0:
            hour_clks.append(np.sum(time_slot_clks[i - 3 : i + 1]))

    hour_clks_df = pd.DataFrame(data=hour_clks)
    hour_clks_df.to_csv('result/test_hour_clks_' + str(budget_para) + '.csv', header=None)
    action_df = pd.DataFrame(data=action_records)
    action_df.to_csv('result/test_action_' + str(budget_para) + '.csv')

    lamda_record_df = pd.DataFrame(data=lamda_record)
    lamda_record_df.to_csv('result/test_lamda_' + str(budget_para) + '.csv')

    print('测试集中：真实曝光数{}, 赢标数{}, 共获得{}个点击, 真实点击数{}, '
          '利润{}, 预算{}, 花费{}, CPM{}, {}'.format(episode_imps, episode_win_imps, episode_clks, episode_real_clks,
                                               episode_profit, budget, episode_spent, episode_spent / episode_win_imps, datetime.datetime.now()))
    test_result_data = []
    test_result_data.append([episode_imps, episode_win_imps, episode_clks, episode_real_clks,
                             episode_profit, budget, episode_spent, episode_spent / episode_win_imps])

    columns = ['real_imps', 'win_imps', 'clks', 'real_clks', 'profit', 'budget', 'spent', 'CPM']
    test_result_data_df = pd.DataFrame(data=test_result_data, columns=columns)
    test_result_data_df.to_csv('result/result_' + str(budget_para) + '.csv')

    return episode_clks

if __name__ == '__main__':
    env = AD_env()
    RL = DRLB([-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08],  # 按照数据集中的“块”计量
             env.action_numbers, env.feature_numbers,
             learning_rate=config['learning_rate'],  # DQN更新公式的学习率
             reward_decay=config['reward_decay'],  # 奖励折扣因子
             e_greedy=config['e_greedy'],  # 贪心算法ε
             replace_target_iter=config['relace_target_iter'],  # 每200步替换一次target_net的参数
             memory_size=config['memory_size'],  # 经验池上限
             batch_size=config['batch_size'],  # 每次更新时从memory里面取多少数据出来，mini-batch
             )

    RewardNet = RewardNet([-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08],  # 按照数据集中的“块”计量
                          1, env.feature_numbers, memory_size=config['memory_size'], batch_size=config['batch_size'], )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        print('-----当前预算条件{}----\n'.format(budget_para[i]))
        test_budget = config['test_budget'] * budget_para[i]
        test_auc_numbers = config['test_auc_num']
        print('\n--------------final test--------------\n')
        run_test(test_budget, test_auc_numbers, budget_para[i])