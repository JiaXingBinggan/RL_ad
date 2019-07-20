# 用于测试数据
from src.compare_rtb.No_reward.DQN.env import AD_env
from src.compare_rtb.No_reward.DQN.RL_brain_for_test import DQN_FOR_TEST
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config

def test_env_threshold(budget, auc_num, budget_para, data_ctr_threshold, env, RL):
    env.build_env(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../../../../../../../data/fm/test_fm_embedding.csv", header=None)

    test_data = test_data.values
    result_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    hour_clks = [0 for i in range(0, 24)]
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

        if current_data_ctr >= data_ctr_threshold:
            bid_nums += 1

            state[2: config['feature_num']] = auc_data[0: config['data_feature_index']]
            state_full = np.array(state, dtype=float)

            state_deep_copy = copy.deepcopy(state_full)
            state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget, state_deep_copy[1] / auc_num

            budget_remain_scale = state[0] / budget
            time_remain_scale = (24 - hour_index) / 24
            # 当后面预算不够但是拍卖数量还多时，应当出价降低，反之可以适当提升
            time_budget_remain_rate = budget_remain_scale / time_remain_scale

            # RL代理根据状态选择动作
            action = RL.choose_best_action(state_deep_copy)
            action = int(action * time_budget_remain_rate)  # 调整出价
            action = action if action <= 300 else 300
            action = action if action > 0 else 1

            # RL采用动作后获得下一个状态的信息以及奖励
            state_, reward, done, is_win = env.step_for_test(auc_data, action, current_data_ctr)

            if is_win:
                hour_clks[int(hour_index)] += current_data_clk
                total_reward_profits += (current_data_ctr * eCPC - auc_data[config['data_marketprice_index']])
                total_reward_clks += current_data_clk
                total_imps += 1
                spent_ += auc_data[config['data_marketprice_index']]

            ctr_action_records.append(
                    [current_data_clk, current_data_ctr, action, auc_data[config['data_marketprice_index']]])

            if done:
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
        else:
            no_bid_hour_clks[int(hour_index)] += current_data_clk

        real_clks += current_data_clk
        real_hour_clks[int(hour_index)] += current_data_clk

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

    hour_clks_array = {'no_bid_hour_clks': no_bid_hour_clks, 'hour_clks': hour_clks, 'real_hour_clks': real_hour_clks,
                       'avg_threshold': data_ctr_threshold}
    hour_clks_df = pd.DataFrame(hour_clks_array)
    hour_clks_df.to_csv('result/test_hour_clks_' + str(budget_para) + '.csv')

    ctr_action_df = pd.DataFrame(data=ctr_action_records)
    ctr_action_df.to_csv('result/test_ctr_action_' + str(budget_para) + '.csv', index=None,
                         header=None)

def to_test(run_model, budget_para):
    env = AD_env()
    RL = DQN_FOR_TEST([action for action in np.arange(1, 301)],  # 按照数据集中的“块”计量
                      env.action_numbers, env.feature_numbers,
                      run_model,
                      )
    for i in range(len(budget_para)):
        print('########测试结果########\n')
        if run_model == 'threshold':
            budget_para = budget_para[i]
            train_pctr_price = pd.read_csv('../../../../transform_precess/20130606_train_ctr_clk.csv', header=None).drop(0,
                                                                                                                   axis=0)
            train_pctr_price.iloc[:, [1, 2]] = train_pctr_price.iloc[:, [1, 2]].astype(float)  # 按列强制类型转换
            ascend_train_pctr_price = train_pctr_price.sort_values(by=1, ascending=False)
            data_ctr_threshold = 0
            print('calculating threshold....\n')
            for i in range(0, len(ascend_train_pctr_price)):
                if np.sum(ascend_train_pctr_price.iloc[:i, 2]) > config['train_budget'] * budget_para:
                    data_ctr_threshold = ascend_train_pctr_price.iloc[i - 1, 1]
                    break
            print(data_ctr_threshold)
            test_budget = config['test_budget'] * budget_para
            test_env_threshold(test_budget, int(config['test_auc_num']), budget_para, data_ctr_threshold, env, RL)