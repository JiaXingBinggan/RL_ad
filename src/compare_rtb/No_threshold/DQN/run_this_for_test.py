# 用于测试数据
from src.compare_rtb.No_threshold.DQN.env_test import AD_env
from src.compare_rtb.No_threshold.DQN.RL_brain_for_test import DQN_FOR_TEST
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config

def test_env(budget, auc_num, budget_para, env, RL):
    env.build_env(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../data/fm/test_fm_embedding.csv", header=None)

    test_total_clks = np.sum(test_data.iloc[:, config['data_clk_index']])
    test_data = test_data.values
    result_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    hour_clks = [0 for i in range(0, 24)]
    no_bid_index = 0
    no_bid_hour_clks = [0 for i in range(0, 24)]
    real_hour_clks = [0 for i in range(0, 24)]

    total_reward_clks = 0
    total_reward_profits = 0
    total_imps = 0
    real_clks = 0
    bid_nums = 0  # 出价次数
    real_imps = 0  # 真实曝光数
    spent_ = 0  # 花费

    is_done = False
    current_with_clk_aucs = 0  # 当前时刻有点击的曝光数量
    current_no_clk_aucs = 0  # 当前时刻没有点击的曝光数量
    current_clk_no_win_aucs = 0  # 当前时刻有点击没赢标的曝光数量
    current_no_clk_no_win_aucs = 0  # 当前时刻没有点击且没赢标的曝光数量
    current_no_clk_win_aucs = 0

    ctr_action_records = []  # 记录模型出价以及真实出价，以及ctr（在有点击数的基础上）
    eCPC = 30000

    for i in range(len(test_data)):
        real_imps += 1

        # auction全部数据
        auc_data = test_data[i: i + 1, :].flatten().tolist()

        # auction所在小时段索引
        hour_index = auc_data[config['data_hour_index']]

        current_data_ctr = auc_data[config['data_pctr_index']]  # 当前数据的ctr，原始为str，应该转为float
        current_data_clk = int(auc_data[config['data_clk_index']])

        bid_nums += 1

        state[2: config['feature_num']] = auc_data[0: config['data_feature_index']]
        state_full = np.array(state, dtype=float)

        state_deep_copy = copy.deepcopy(state_full)
        state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget, state_deep_copy[1] / auc_num

        # RL代理根据状态选择动作
        action = RL.choose_best_action(state_deep_copy)
        action = action if action <= 300 else 300
        action = action if action > 0 else 1

        # 获得remainClks和remainBudget的比例，以及punishRate
        remainClkRate = (test_total_clks - real_clks) / test_total_clks
        remainBudgetRate = state[0] / budget
        punishRate = remainClkRate / remainBudgetRate

        # 记录当前时刻有点击没赢标的曝光数量以及punishNoWinRate
        if current_data_clk == 1:
            current_with_clk_aucs += 1
            if action < auc_data[config['data_marketprice_index']]:
                current_clk_no_win_aucs += 1
        else:
            current_no_clk_aucs += 1
            if action > auc_data[config['data_marketprice_index']]:
                current_no_clk_win_aucs += 1
            else:
                current_no_clk_no_win_aucs += 1

        temp_adjust_rate = (current_clk_no_win_aucs / current_with_clk_aucs) if current_with_clk_aucs > 0 else 1
        punishNoWinRate = (1 - temp_adjust_rate) if temp_adjust_rate != 1 else 1

        # 记录基础鼓励值baseEncourage，及鼓励比例encourageRate
        baseEncourage = auc_data[config['data_marketprice_index']]
        encourageRate = (1 - current_no_clk_no_win_aucs / current_no_clk_aucs) if current_no_clk_aucs > 0 else 0
        encourageNoClkNoWin = (baseEncourage / encourageRate) if encourageRate > 0 else 1

        # RL采用动作后获得下一个状态的信息以及奖励
        state_, reward, done, is_win = env.step_profit_for_test(auc_data, action, current_data_ctr,
                                                                punishRate, punishNoWinRate, encourageNoClkNoWin)

        if is_win:
            hour_clks[int(hour_index)] += current_data_clk
            total_reward_profits += (current_data_ctr * eCPC - auc_data[config['data_marketprice_index']])
            total_reward_clks += current_data_clk
            total_imps += 1
            spent_ += auc_data[config['data_marketprice_index']]

        ctr_action_records.append(
            [current_data_clk, current_data_ctr, action, auc_data[config['data_marketprice_index']]])

        if done:
            no_bid_index = i
            is_done = True
            if state_[0] < 0:
                spent = budget
            else:
                spent = budget - state_[0]
            cpm = (spent / total_imps) if total_imps > 0 else 0
            result_array.append(
                [total_reward_clks, real_imps, bid_nums, total_imps, budget, spent, cpm, real_clks,
                 total_reward_profits])
            break

        if bid_nums % 100000 == 0:
            now_spent = budget - state_[0]
            if total_imps != 0:
                now_cpm = now_spent / total_imps
            else:
                now_cpm = 0
            print('当前: 真实曝光数{}, 出价数{}, 赢标数{}, 当前利润{}, 当前点击数{}, 真实点击数{}, 预算{}, 花费{}, CPM{}\t{}'.format(
                real_imps, bid_nums, total_imps, total_reward_profits, total_reward_clks,
                real_clks, budget, now_spent, now_cpm, datetime.datetime.now()))

        real_clks += current_data_clk
        real_hour_clks[int(hour_index)] += current_data_clk

        no_bid_index = i

    if not is_done:
        result_array.append(
            [total_reward_clks, real_imps, bid_nums, total_imps, budget, spent_, spent_ / total_imps, real_clks,
             total_reward_profits])
    print('\n测试集中: 真实曝光数{}，出价数{}, 赢标数{}, 总点击数{}, '
          '真实点击数{}, 预算{}, 总花费{}, CPM{}，总利润{}\n'.format(result_array[0][1], result_array[0][2],
                                                       result_array[0][3], result_array[0][0], result_array[0][7],
                                                       result_array[0][4],
                                                       result_array[0][5], result_array[0][6], result_array[0][8]))
    result_df = pd.DataFrame(data=result_array,
                             columns=['clks', 'real_imps', 'bids', 'imps(wins)', 'budget', 'spent', 'cpm', 'real_clks',
                                      'profits'])
    result_df.to_csv('result/result_' + str(budget_para) + '.txt')

    if no_bid_index != len(test_data) - 1:
        for k in range(no_bid_index + 1, len(test_data)):  # 记录未参与投标的点击数（漏掉的）
            auc_data = test_data[k: k + 1, :].flatten().tolist()
            hour_index = int(auc_data[config['data_hour_index']])
            no_bid_hour_clks[hour_index] += int(auc_data[config['data_clk_index']])
            real_hour_clks[int(hour_index)] += int(auc_data[config['data_clk_index']])

    hour_clks_array = {'no_bid_hour_clks': no_bid_hour_clks, 'hour_clks': hour_clks, 'real_hour_clks': real_hour_clks}
    hour_clks_df = pd.DataFrame(hour_clks_array)
    hour_clks_df.to_csv('result/test_hour_clks_' + str(budget_para) + '.csv')

    ctr_action_df = pd.DataFrame(data=ctr_action_records)
    ctr_action_df.to_csv('result/test_ctr_action_' + str(budget_para) + '.csv', index=None,
                         header=None)

def to_test(budget_para):
    env = AD_env()
    RL = DQN_FOR_TEST([action for action in np.arange(1, 301)],  # 按照数据集中的“块”计量
                      env.action_numbers, env.feature_numbers,
                      )
    for i in range(len(budget_para)):
        print('########测试结果########\n')
        budget_para = budget_para[i]
        test_budget = config['test_budget'] * budget_para
        test_env(test_budget, int(config['test_auc_num']), budget_para, env, RL)