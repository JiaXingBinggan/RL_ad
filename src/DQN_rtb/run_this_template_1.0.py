from src.DQN_rtb.env_test import AD_env
from src.DQN_rtb.RL_brain import DQN
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config

def delta_time(time_index):
    if time_index >= 1 and time_index <= 9:
        time_clk_rate = 55/328
    else:
        time_clk_rate = 273/328
    return time_clk_rate

def run_env(budget, auc_num, budget_para):
    env.build_env(budget, auc_num) # 参数为训练集的(预算， 预期展示次数)
    # 训练
    step = 0
    print('data loading\n')
    train_data = pd.read_csv("../../data/fm/train_fm_embedding.csv", header=None)
    train_data.iloc[:, config['data_hour_index']] = train_data.iloc[:, config['data_hour_index']].astype(int) # 将时间序列设置为Int类型
    train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs_1.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率

    # 获取大于ctr阈值的数据索引
    hour_index = train_data.iloc[:, config['data_hour_index']]
    compare_ctr_index = train_data[train_data.iloc[:, config['data_pctr_index']] > train_avg_ctr[hour_index]].index.values.tolist()

    train_total_clks = np.sum(train_data.iloc[:, config['data_clk_index']])
    records_array = [] # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    eCPC = 30000 # 每次点击花费
    for episode in range(config['train_episodes']):
        # 初始化状态
        state = env.reset(budget, auc_num) # 参数为训练集的(预算， 总展示次数)

        print('第{}轮'.format(episode + 1))
        hour_clks = [0 for i in range(0, 24)] # 记录每个小时获得点击数
        no_bid_hour_clks = [0 for i in range(0, 24)] # 记录被过滤掉但没有投标的点击数
        real_hour_clks = [0 for i in range(0, 24)] # 记录数据集中真实点击数

        total_reward_clks = 0
        total_reward_profits = 0
        total_imps = 0
        real_clks = 0 # 数据集真实点击数（到目前为止，或整个数据集）
        bid_nums = 0 # 出价次数
        real_imps = 0 # 真实曝光数

        current_with_clk_aucs = 0 # 当前时刻有点击的曝光数量
        current_no_clk_aucs = 0 # 当前时刻没有点击的曝光数量
        current_clk_no_win_aucs = 0 # 当前时刻有点击没赢标的曝光数量
        current_no_clk_no_win_aucs = 0 # 当前时刻没有点击且没赢标的曝光数量
        current_no_clk_win_aucs = 0

        ctr_action_records = [] # 记录模型出价以及真实出价，以及ctr（在有点击数的基础上）

        for i in range(len(train_data)):
            real_imps += 1

            auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()

            # auction所在小时段索引
            hour_index = auc_data[config['data_hour_index']]

            state[2: config['feature_num']] = auc_data[0: config['data_feature_index']]
            state_full = np.array(state, dtype=float)
            # 预算以及剩余拍卖数量缩放，避免因预算及拍卖数量数值过大引起神经网络性能不好
            # 执行深拷贝，防止修改原始数据
            state_deep_copy = copy.deepcopy(state_full)
            state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget,  state_deep_copy[1] / auc_num

            current_data_ctr = auc_data[config['data_pctr_index']] # 当前数据的ctr，原始为str，应该转为float
            current_data_clk = int(auc_data[config['data_clk_index']])

            budget_remain_scale = state[0] / budget
            time_remain_scale = (24 - hour_index) / 24
            time_clk_rate = delta_time(int(hour_index))
            # 当后面预算不够但是拍卖数量还多时，应当出价降低，反之可以适当提升
            time_budget_remain_rate = time_clk_rate * budget_remain_scale / time_remain_scale
            if current_data_ctr >= train_avg_ctr[int(hour_index)]: # 乘以1/2
                bid_nums += 1

                # RL代理根据状态选择动作
                action, mark = RL.choose_action(state_deep_copy, current_data_ctr)
                action = int(action * time_budget_remain_rate) # 直接取整是否妥当？
                action = action if action <= 300 else 300
                current_mark = mark

                # 获取剩下的数据
                # 下一个状态的特征（除去预算、剩余拍卖数量）
                if compare_ctr_index.index(i) != len(compare_ctr_index) - 1:
                    next_index = compare_ctr_index[compare_ctr_index.index(i) + 1]
                    auc_data_next = train_data.iloc[next_index: next_index + 1, :].values.flatten().tolist()[
                                    0: config['data_feature_index']]
                else:
                    auc_data_next = [0 for i in range(config['state_feature_num'])]

                # 获得remainClks和remainBudget的比例，以及punishRate
                remainClkRate = np.sum(train_data.iloc[i+1 :, config['data_clk_index']]) / train_total_clks
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
                # 下一个状态包括了下一步的预算、剩余拍卖数量以及下一条数据的特征向量
                state_, reward, done, is_win = env.step_profit(auc_data, action, auc_data_next, current_data_ctr,
                                                               punishRate, punishNoWinRate, encourageNoClkNoWin)

                # RL代理将 状态-动作-奖励-下一状态 存入经验池
                # 深拷贝
                state_next_deep_copy = copy.deepcopy(state_)
                state_next_deep_copy[0], state_next_deep_copy[1] = state_next_deep_copy[0] / budget, state_next_deep_copy[1] / auc_num
                RL.store_transition(state_deep_copy.tolist(), action, reward, state_next_deep_copy)

                if is_win:
                    hour_clks[int(hour_index)] += current_data_clk
                    total_reward_clks += current_data_clk
                    total_reward_profits += (current_data_ctr * eCPC - auc_data[config['data_marketprice_index']])
                    total_imps += 1

                if current_data_clk == 1:
                    ctr_action_records.append([current_data_clk, current_data_ctr, current_mark, action, auc_data[config['data_marketprice_index']]])
                else:
                    ctr_action_records.append([current_data_clk, current_data_ctr, current_mark, action, auc_data[config['data_marketprice_index']]])

                # 当经验池数据达到一定量后再进行学习
                if (step > config['batch_size']) and (step % 16 == 0): # 控制更新速度
                    RL.learn()

                # 将下一个state_变为 下次循环的state
                state = state_

                # 如果终止（满足一些条件），则跳出循环
                if done:
                    if state_[0] < 0:
                        spent = budget
                    else:
                        spent = budget - state_[0]
                    cpm = spent / total_imps
                    records_array.append([total_reward_clks, real_imps, bid_nums, total_imps, budget, spent, cpm, real_clks,
                                          total_reward_profits])
                    break
                elif compare_ctr_index.index(i) == len(compare_ctr_index) - 1:
                    if state_[0] < 0:
                        spent = budget
                    else:
                        spent = budget - state_[0]
                    cpm = spent / total_imps
                    records_array.append([total_reward_clks, real_imps, bid_nums, total_imps, budget, spent, cpm, real_clks,
                                          total_reward_profits])
                    break

                step += 1

                if bid_nums % 10000 == 0:
                    now_spent = budget - state_[0]
                    if total_imps != 0:
                        now_cpm = now_spent / total_imps
                    else:
                        now_cpm = 0
                    print('episode {}: 真实曝光数{}, 出价数{}, 赢标数{}, 当前利润{}, 当前点击数{}, 真实点击数{}, 预算{}, 花费{}, CPM{}\t{}'.format(episode + 1, real_imps,
                                                                      bid_nums,total_imps,total_reward_profits,total_reward_clks, real_clks,
                                                                      budget,now_spent,now_cpm,datetime.datetime.now()))
            else:
                no_bid_hour_clks[int(hour_index)] += current_data_clk


            real_clks += current_data_clk
            real_hour_clks[int(hour_index)] += current_data_clk

        RL.control_epsilon() # 每轮，逐渐增加epsilon，增加行为的利用性
        RL.store_para('template') # 每一轮存储一次参数

        # 出现提前终止，done=False的结果展示
        # 如果没有处理，会出现index out
        if len(records_array) == 0:
            records_array_tmp = [[0 for i in range(9)]]
            episode_record = records_array_tmp[0]
        else:
            episode_record = records_array[episode]
        print('\n第{}轮: 真实曝光数{}, 出价次数{}, 赢标数{}, 总利润{}, 总点击数{}, 真实点击数{}, 预算{}, 总花费{}, CPM{}\n'.format(episode + 1,
                          episode_record[1],episode_record[2],episode_record[3],episode_record[8], episode_record[0],episode_record[7],
                          episode_record[4],episode_record[5],episode_record[6]))

        ctr_action_df = pd.DataFrame(data=ctr_action_records)
        ctr_action_df.to_csv('../../result/DQN/profits/train_ctr_action_' + str(budget_para) + '.csv', index=None, header=None)

        hour_clks_array = {'no_bid_hour_clks': no_bid_hour_clks, 'hour_clks': hour_clks, 'real_hour_clks': real_hour_clks, 'avg_threshold': train_avg_ctr}
        hour_clks_df = pd.DataFrame(hour_clks_array)
        hour_clks_df.to_csv('../../result/DQN/profits/train_hour_clks_' + str(budget_para) + '.csv')

        if (episode + 1) % 10 == 0:
            print('\n########当前测试结果########\n')
            test_env(config['test_budget']*config['budget_para'][0], int(config['test_auc_num']), config['budget_para'][0])

    print('训练结束\n')

    records_df = pd.DataFrame(data=records_array,
                              columns=['clks', 'real_imps', 'bids', 'imps(wins)', 'budget', 'spent', 'cpm', 'real_clks', 'profits'])
    records_df.to_csv('../../result/DQN/profits/train_' + str(budget_para) + '.txt')

def test_env(budget, auc_num, budget_para):
    env.build_env(budget, auc_num) # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num) # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../data/fm/test_fm_embedding.csv", header=None)
    train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs_1.csv", header=None).iloc[:,1].values  # 用前一天预测后一天中每个时段的平均点击率

    test_total_clks = int(np.sum(test_data.iloc[:, config['data_clk_index']]))
    result_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    hour_clks = [0 for i in range(0, 24)]
    no_bid_hour_clks = [0 for i in range(0, 24)]
    real_hour_clks = [0 for i in range(0, 24)]

    total_reward_clks = 0
    total_reward_profits = 0
    total_imps = 0
    real_clks = 0
    bid_nums = 0 # 出价次数
    real_imps = 0 # 真实曝光数

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
        auc_data = test_data.iloc[i: i + 1, :].values.flatten().tolist()

        # auction所在小时段索引
        hour_index = auc_data[config['data_hour_index']]

        state[2: config['feature_num']] = auc_data[0: config['data_feature_index']]
        state_full = np.array(state, dtype=float)

        state_deep_copy = copy.deepcopy(state_full)
        state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget, state_deep_copy[1] / auc_num

        current_data_ctr = auc_data[config['data_pctr_index']]  # 当前数据的ctr，原始为str，应该转为float
        current_data_clk = int(auc_data[config['data_clk_index']])

        budget_remain_scale = state[0] / budget
        time_remain_scale = (24 - hour_index) / 24
        time_clk_rate = delta_time(int(hour_index))
        # 当后面预算不够但是拍卖数量还多时，应当出价降低，反之可以适当提升
        time_budget_remain_rate = time_clk_rate * budget_remain_scale / time_remain_scale

        if current_data_ctr >= train_avg_ctr[int(hour_index)]:
            bid_nums += 1

            # RL代理根据状态选择动作
            action = RL.choose_best_action(state_deep_copy)
            action = int(action * time_budget_remain_rate) # 调整出价
            action = action if action <= 300 else 300

            # 获得remainClks和remainBudget的比例，以及punishRate
            remainClkRate = np.sum(test_data.iloc[i + 1:, config['data_clk_index']]) / test_total_clks
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

            if current_data_clk == 1:
                ctr_action_records.append([current_data_clk, current_data_ctr, action, auc_data[config['data_marketprice_index']]])
            else:
                ctr_action_records.append([current_data_clk,current_data_ctr, action, auc_data[config['data_marketprice_index']]])

            if done:
                if state_[0] < 0:
                    spent = budget
                else:
                    spent = budget - state_[0]
                cpm = (spent / total_imps) if total_imps > 0 else 0
                result_array.append(
                    [total_reward_clks, real_imps, bid_nums, total_imps, budget, spent, cpm, real_clks, total_reward_profits])
                break

            if bid_nums % 10000 == 0:
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

    if len(result_array) == 0:
        result_array = [[0 for i in range(9)]]
    print('\n测试集中: 真实曝光数{}，出价数{}, 赢标数{}, 总点击数{}, '
          '真实点击数{}, 预算{}, 总花费{}, CPM{}，总利润{}\n'.format(result_array[0][1], result_array[0][2],
                                  result_array[0][3],result_array[0][0], result_array[0][7], result_array[0][4],
                                  result_array[0][5], result_array[0][6], result_array[0][8]))
    result_df = pd.DataFrame(data=result_array, columns=['clks', 'real_imps', 'bids', 'imps(wins)', 'budget', 'spent', 'cpm', 'real_clks', 'profits'])
    result_df.to_csv('../../result/DQN/profits/result_' + str(budget_para) + '.txt')

    hour_clks_array = {'no_bid_hour_clks': no_bid_hour_clks, 'hour_clks': hour_clks, 'real_hour_clks': real_hour_clks, 'avg_threshold': train_avg_ctr}
    hour_clks_df = pd.DataFrame(hour_clks_array)
    hour_clks_df.to_csv('../../result/DQN/profits/test_hour_clks_' + str(budget_para) + '.csv')

    ctr_action_df = pd.DataFrame(data=ctr_action_records)
    ctr_action_df.to_csv('../../result/DQN/profits/test_ctr_action_' + str(budget_para) + '.csv', index=None, header=None)

if __name__ == '__main__':
    env = AD_env()
    RL = DQN([action for action in np.arange(1, 301)], # 按照数据集中的“块”计量
              env.action_numbers, env.feature_numbers,
              learning_rate=config['learning_rate'], # DQN更新公式的学习率
              reward_decay=config['reward_decay'], # 奖励折扣因子
              e_greedy=config['e_greedy'], # 贪心算法ε
              replace_target_iter=config['relace_target_iter'], # 每200步替换一次target_net的参数
              memory_size=config['memory_size'], # 经验池上限
              batch_size=config['batch_size'], # 每次更新时从memory里面取多少数据出来，mini-batch
              )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        print('-----当前预算条件{}----\n'.format(budget_para[i]))
        train_budget, train_auc_numbers = config['train_budget'], int(config['train_auc_num'])
        test_budget, test_auc_numbers = config['test_budget']*budget_para[i], int(config['test_auc_num'])
        run_env(train_budget, train_auc_numbers, budget_para[i])
        print('########测试结果########\n')
        test_env(test_budget, test_auc_numbers, budget_para[i])