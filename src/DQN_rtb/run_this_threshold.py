from src.DQN_rtb.env_test import AD_env
from src.DQN_rtb.RL_brain import DQN
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config

def run_env(budget, auc_num, budget_para, data_ctr_threshold):
    env.build_env(budget, auc_num)  # 参数为训练集的(预算， 预期展示次数)
    # 训练
    step = 0
    print('data loading\n')
    train_data = pd.read_csv("../../data/fm/train_fm_embedding.csv", header=None)
    train_data.iloc[:, config['data_hour_index']] = train_data.iloc[:, config['data_hour_index']].astype(
        int)  # 将时间序列设置为Int类型

    # 获取大于ctr阈值的数据索引
    compare_ctr_index = train_data[
        train_data.iloc[:, config['data_pctr_index']] >= data_ctr_threshold].index.values.tolist()

    train_total_clks = np.sum(train_data.iloc[:, config['data_clk_index']])
    records_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    test_records_array = []
    eCPC = 30000  # 每次点击花费
    for episode in range(config['train_episodes']):
        # 初始化状态
        state = env.reset(budget, auc_num)  # 参数为训练集的(预算， 总展示次数)

        print('第{}轮'.format(episode + 1))
        hour_clks = [0 for i in range(0, 24)]  # 记录每个小时获得点击数
        no_bid_hour_clks = [0 for i in range(0, 24)]  # 记录被过滤掉但没有投标的点击数
        real_hour_clks = [0 for i in range(0, 24)]  # 记录数据集中真实点击数

        is_done = False
        spent_ = 0
        total_reward_clks = 0
        total_reward_profits = 0
        total_imps = 0
        real_clks = 0  # 数据集真实点击数（到目前为止，或整个数据集）
        bid_nums = 0  # 出价次数
        real_imps = 0  # 真实曝光数

        current_with_clk_aucs = 0  # 当前时刻有点击的曝光数量
        current_no_clk_aucs = 0  # 当前时刻没有点击的曝光数量
        current_clk_no_win_aucs = 0  # 当前时刻有点击没赢标的曝光数量
        current_no_clk_no_win_aucs = 0  # 当前时刻没有点击且没赢标的曝光数量
        current_no_clk_win_aucs = 0

        ctr_action_records = []  # 记录模型出价以及真实出价，以及ctr（在有点击数的基础上）

        for i in range(len(train_data)):
            real_imps += 1

            auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()

            # auction所在小时段索引
            hour_index = auc_data[config['data_hour_index']]

            current_data_ctr = auc_data[config['data_pctr_index']]  # 当前数据的ctr，原始为str，应该转为float
            current_data_clk = int(auc_data[config['data_clk_index']])

            if current_data_ctr >= data_ctr_threshold:  # 乘以1/2
                bid_nums += 1

                state[2: config['feature_num']] = auc_data[0: config['data_feature_index']]
                state_full = np.array(state, dtype=float)
                # 预算以及剩余拍卖数量缩放，避免因预算及拍卖数量数值过大引起神经网络性能不好
                # 执行深拷贝，防止修改原始数据
                state_deep_copy = copy.deepcopy(state_full)
                state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget, state_deep_copy[1] / auc_num

                budget_remain_scale = state[0] / budget
                time_remain_scale = (24 - hour_index) / 24
                # 当后面预算不够但是拍卖数量还多时，应当出价降低，反之可以适当提升
                time_budget_remain_rate = budget_remain_scale / time_remain_scale

                # RL代理根据状态选择动作
                action, mark = RL.choose_action(state_deep_copy, current_data_ctr)
                action = int(action * time_budget_remain_rate)  # 直接取整是否妥当？
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
                remainClkRate = np.sum(train_data.iloc[i + 1:, config['data_clk_index']]) / train_total_clks
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
                state_next_deep_copy[0], state_next_deep_copy[1] = state_next_deep_copy[0] / budget, \
                                                                   state_next_deep_copy[1] / auc_num
                RL.store_transition(state_deep_copy.tolist(), action, reward, state_next_deep_copy)

                if is_win:
                    spent_ += auc_data[config['data_marketprice_index']]
                    hour_clks[int(hour_index)] += current_data_clk
                    total_reward_clks += current_data_clk
                    total_reward_profits += (current_data_ctr * eCPC - auc_data[config['data_marketprice_index']])
                    total_imps += 1

                if current_data_clk == 1:
                    ctr_action_records.append([current_data_clk, current_data_ctr, current_mark, action,
                                               auc_data[config['data_marketprice_index']]])
                else:
                    ctr_action_records.append([current_data_clk, current_data_ctr, current_mark, action,
                                               auc_data[config['data_marketprice_index']]])

                # 当经验池数据达到一定量后再进行学习
                if (step > config['batch_size']) and (step % 16 == 0):  # 控制更新速度
                    RL.learn()

                # 将下一个state_变为 下次循环的state
                state = state_

                # 如果终止（满足一些条件），则跳出循环
                if done:
                    is_done = True
                    if state_[0] < 0:
                        spent = budget
                    else:
                        spent = budget - state_[0]
                    cpm = spent / total_imps
                    records_array.append(
                        [total_reward_clks, real_imps, bid_nums, total_imps, budget, spent, cpm, real_clks,
                         total_reward_profits])
                    break

                step += 1

                if bid_nums % 10000 == 0:
                    now_spent = budget - state_[0]
                    if total_imps != 0:
                        now_cpm = now_spent / total_imps
                    else:
                        now_cpm = 0
                    print('episode {}: 真实曝光数{}, 出价数{}, 赢标数{}, 当前利润{}, 当前点击数{}, 真实点击数{}, 预算{}, 花费{}, CPM{}\t{}'.format(
                        episode + 1, real_imps,
                        bid_nums, total_imps, total_reward_profits, total_reward_clks, real_clks,
                        budget, now_spent, now_cpm, datetime.datetime.now()))
            else:
                no_bid_hour_clks[int(hour_index)] += current_data_clk

            real_clks += current_data_clk
            real_hour_clks[int(hour_index)] += current_data_clk

        if not is_done:
            records_array.append([total_reward_clks, real_imps, bid_nums, total_imps, budget, spent_, spent_ / total_imps, real_clks,
             total_reward_profits])
        RL.control_epsilon()  # 每轮，逐渐增加epsilon，增加行为的利用性

        # 出现提前终止，done=False的结果展示
        # 如果没有处理，会出现index out
        if len(records_array) == 0:
            records_array_tmp = [[0 for i in range(9)]]
            episode_record = records_array_tmp[0]
        else:
            episode_record = records_array[episode]
        print('\n第{}轮: 真实曝光数{}, 出价次数{}, 赢标数{}, 总利润{}, 总点击数{}, 真实点击数{}, 预算{}, 总花费{}, CPM{}\n'.format(episode + 1,
                                                                                                    episode_record[1],
                                                                                                    episode_record[2],
                                                                                                    episode_record[3],
                                                                                                    episode_record[8],
                                                                                                    episode_record[0],
                                                                                                    episode_record[7],
                                                                                                    episode_record[4],
                                                                                                    episode_record[5],
                                                                                                    episode_record[6]))

        ctr_action_df = pd.DataFrame(data=ctr_action_records)
        ctr_action_df.to_csv('../../result/DQN/profits/train_ctr_action_' + str(budget_para) + '.csv', index=None,
                             header=None)

        hour_clks_array = {'no_bid_hour_clks': no_bid_hour_clks, 'hour_clks': hour_clks,
                           'real_hour_clks': real_hour_clks, 'avg_threshold': data_ctr_threshold}
        hour_clks_df = pd.DataFrame(hour_clks_array)
        hour_clks_df.to_csv('../../result/DQN/profits/train_hour_clks_' + str(budget_para) + '.csv')

        if (episode + 1) % 10 == 0:
            print('\n########当前测试结果########\n')
            test_result = test_env(config['test_budget'] * budget_para, int(config['test_auc_num']), budget_para,data_ctr_threshold)
            test_records_array.append(test_result)

        test_clks_record = np.array(test_records_array)[:, 0]
        test_clks_array = test_clks_record.astype(np.int).tolist()

        max = RL.para_store_iter(test_clks_array)
        if max == test_clks_array[len(test_clks_array) - 1:len(test_clks_array)][0]:
            print('最优参数已存储')
            RL.store_para('threshold')  # 存储最大值

    print('训练结束\n')

    records_df = pd.DataFrame(data=records_array,
                              columns=['clks', 'real_imps', 'bids', 'imps(wins)', 'budget', 'spent', 'cpm', 'real_clks',
                                       'profits'])
    records_df.to_csv('../../result/DQN/profits/train_' + str(budget_para) + '.txt')

    test_records_df = pd.DataFrame(data=test_records_array, columns=['clks', 'real_imps', 'bids',
                                                                     'imps(wins)', 'budget', 'spent',
                                                                     'cpm', 'real_clks', 'profits'])
    test_records_df.to_csv('../../result/DQN/profits/episode_test_' + str(budget_para) + '.txt')

def test_env(budget, auc_num, budget_para, data_ctr_threshold):
    env.build_env(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num)  # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../data/fm/test_fm_embedding.csv", header=None)

    test_total_clks = int(np.sum(test_data.iloc[:, config['data_clk_index']]))
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
                spent_ += auc_data[config['data_marketprice_index']]

            if current_data_clk == 1:
                ctr_action_records.append(
                    [current_data_clk, current_data_ctr, action, auc_data[config['data_marketprice_index']]])
            else:
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
    result_df.to_csv('../../result/DQN/profits/result_' + str(budget_para) + '.txt')

    hour_clks_array = {'no_bid_hour_clks': no_bid_hour_clks, 'hour_clks': hour_clks, 'real_hour_clks': real_hour_clks,
                       'avg_threshold': data_ctr_threshold}
    hour_clks_df = pd.DataFrame(hour_clks_array)
    hour_clks_df.to_csv('../../result/DQN/profits/test_hour_clks_' + str(budget_para) + '.csv')

    ctr_action_df = pd.DataFrame(data=ctr_action_records)
    ctr_action_df.to_csv('../../result/DQN/profits/test_ctr_action_' + str(budget_para) + '.csv', index=None,
                         header=None)

    result_ = [total_reward_clks, real_imps, bid_nums, total_imps, budget, spent_, spent_ / total_imps, real_clks,
               total_reward_profits]
    return result_

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
              # output_graph=True # 是否输出tensorboard文件
              )

    '''
    把pctr降序排列，根据预算，使得处于某阈值以上的市场价格之和小于此预算，则起得过滤的作用
    '''
    train_pctr_price = pd.read_csv('../../transform_precess/20130606_train_ctr_clk.csv', header=None).drop(0, axis=0)
    train_pctr_price.iloc[:, [1, 2]] = train_pctr_price.iloc[:, [1, 2]].astype(float) # 按列强制类型转换
    ascend_train_pctr_price = train_pctr_price.sort_values(by=1, ascending=False)
    data_ctr_threshold = 0
    data_num = 0
    print('calculating threshold....\n')

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        for k in range(0, len(ascend_train_pctr_price)):
            if np.sum(ascend_train_pctr_price.iloc[:k, 2]) > (config['train_budget'] * budget_para[i]):
                data_ctr_threshold = ascend_train_pctr_price.iloc[k - 1, 1]
                data_num = k
                break
        print(data_ctr_threshold)

        train_budget = config['train_budget'] * budget_para[i]
        test_budget = config['test_budget'] * budget_para[i]
        run_env(train_budget, data_num, budget_para[i], data_ctr_threshold)
        print('########测试结果########\n')
        test_env(test_budget, config['test_auc_num'], budget_para[i], data_ctr_threshold)
