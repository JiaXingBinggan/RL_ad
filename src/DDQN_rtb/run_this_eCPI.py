from src.DDQN_rtb.env import AD_env
from src.DDQN_rtb.RL_brain import DoubleDQN
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config

def run_env(budget, auc_num, e_greedy, budget_para):
    env.build_env(budget, auc_num) # 参数为训练集的(预算， 总展示次数)
    # 训练
    step = 0
    print('data loading\n')
    train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
    embedding_v = pd.read_csv("../../data/fm/embedding_v.csv", header=None)
    train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop([0], axis=0).iloc[:, 1].values # 读取训练数据集中每条数据的pctr
    train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:, 1].values # 每个时段的平均点击率

    records_array = [] # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    for episode in range(config['train_episodes']):
        # 初始化状态
        state = env.reset(budget, auc_num) # 参数为训练集的(预算， 总展示次数)
        # 此处的循环为训练数据的长度
        # 状态初始化为预算及拍卖数量，在循环内加上拍卖向量值

        # 重置epsilon
        RL.reset_epsilon(0.9)

        print('第{}轮'.format(episode + 1))
        total_reward = 0
        total_reward_clks = 0
        total_imps = 0
        real_clks = 0 # 数据集真实点击数（到目前为止，或整个数据集）
        for i in range(len(train_data)):
            # auction全部数据
            # random_index = np.random.randint(0, len(train_data))
            # auc_data = train_data.iloc[random_index: random_index + 1, :].values.flatten().tolist()
            auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()
            # auction所在小时段索引
            hour_index = auc_data[18]

            feature_data = [train_ctr[i]] # ctr特征
            # auction特征（除去click，payprice, hour）
            for feat in auc_data[0: 16]:
                feature_data += embedding_v.iloc[feat, :].values.tolist() # 获取对应特征的隐向量

            state[2: 163] = feature_data
            state_full = np.array(state, dtype=float)
            # 预算以及剩余拍卖数量缩放，避免因预算及拍卖数量数值过大引起神经网络性能不好
            # 执行深拷贝，防止修改原始数据
            state_deep_copy = copy.deepcopy(state_full)
            state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget,  state_deep_copy[1] / auc_num

            current_data_ctr = float(train_ctr[i]) # 当前数据的ctr，原始为str，应该转为float

            # 下一个状态的特征（除去预算、剩余拍卖数量）
            auc_data_next = train_data.iloc[i + 1: i + 2, :].values.flatten().tolist()[0: 16]
            next_feature_data = [train_ctr[i+1]]
            for feat_next in auc_data_next:
                next_feature_data += embedding_v.iloc[feat_next, :].values.tolist()
            auc_data_next = np.array(next_feature_data, dtype=float).tolist()

            if current_data_ctr >= train_avg_ctr[int(hour_index)]:

                # RL代理根据状态选择动作
                action = RL.choose_action(state_deep_copy, current_data_ctr, e_greedy)  # 1*17维,第三个参数为epsilon

                # RL采用动作后获得下一个状态的信息以及奖励
                # 下一个状态包括了下一步的预算、剩余拍卖数量以及下一条数据的特征向量
                state_, reward, done, is_win = env.step_eCPI(auc_data, action, auc_data_next)
                # RL代理将 状态-动作-奖励-下一状态 存入经验池
            else:
                action = 0 # 出价为0，即不参与竞标
                state_, reward, done, is_win = env.step_eCPI(auc_data, action, auc_data_next)
            # 深拷贝
            state_next_deep_copy = copy.deepcopy(state_)
            state_next_deep_copy[0], state_next_deep_copy[1] = state_next_deep_copy[0] / budget, state_next_deep_copy[1] / auc_num
            RL.store_transition(state_deep_copy.tolist(), action, reward, state_next_deep_copy)
            if is_win:
                total_reward_clks += reward
                total_imps += 1

            real_clks += int(auc_data[16])
            # 当经验池数据达到一定量后再进行学习
            if (step > 1024) and (step % 4 == 0):
                RL.learn()

            # 将下一个state_变为 下次循环的state
            state = state_

            if i % 1000 == 0:
                now_spent = budget - state_[0]
                if total_imps != 0:
                    now_cpm = now_spent / total_imps
                else:
                    now_cpm = 0
                print('episode {}: 出价数{}, 赢标数{}, 当前点击数{}, 真实点击数{}, 预算{}, 花费{}, CPM{}\t{}'.format(episode, i,
                                                                                     total_imps, total_reward_clks, real_clks,
                                                                                     budget, now_spent, now_cpm, datetime.datetime.now()))
            # 如果终止（满足一些条件），则跳出循环
            if done:
                if state_[0] < 0:
                    spent = budget
                else:
                    spent = budget - state_[0]
                cpm = spent / total_imps
                records_array.append([total_reward_clks, i, total_imps, budget, spent, cpm, real_clks])
                break
            step += 1

        # 出现提前终止，done=False的结果展示
        # 如果没有处理，会出现index out
        if len(records_array) == 0:
            records_array_tmp = [[0 for i in range(7)]]
            episode_record = records_array_tmp[0]
        else:
            episode_record = records_array[episode]
        print('\n第{}轮: 出价数{}, 赢标数{}, 总点击数{}, 真实点击数{}, 预算{}, 总花费{}, CPM{}\n'.format(episode+1,
                    episode_record[1], episode_record[2], episode_record[0], episode_record[6], episode_record[3], episode_record[4],
                                                                              episode_record[5]))
    print('训练结束\n')

    records_df = pd.DataFrame(data=records_array, columns=['clks', 'bids', 'imps(wins)', 'budget', 'spent', 'cpm', 'real_clks'])
    records_df.to_csv('../../result/DDQN_train_' + str(budget_para) + '.txt')

def test_env(budget, auc_num, budget_para):
    env.build_env(budget, auc_num) # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num) # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../data/fm/test_fm.csv", header=None)
    test_ctr = pd.read_csv("../../data/fm/test_ctr_pred.csv", header=None).iloc[:, 1].values  # 读取测试数据集中每条数据的pctr
    embedding_v = pd.read_csv("../../data/fm/embedding_v.csv", header=None)
    test_avg_ctr = pd.read_csv("../../transform_precess/test_avg_ctrs.csv", header=None).iloc[:,1].values  # 测试集中每个时段的平均点击率

    result_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    hour_clks = [0 for i in range(0, 24)]
    real_hour_clks = [0 for i in range(0, 24)]

    total_reward_clks = 0
    total_imps = 0
    real_clks = 0
    for i in range(len(test_data)):
        if i == 0:
            continue
        # auction全部数据
        auc_data = test_data.iloc[i: i + 1, :].values.flatten().tolist()

        # auction所在小时段索引
        hour_index = auc_data[18]

        feature_data = [test_ctr[i]] # ctr特征
        # 二维矩阵转一维，用flatten函数
        # auction特征（除去click，payprice, hour）
        for feat in auc_data[0: 16]:
            feature_data += embedding_v.iloc[feat, :].values.tolist()

        state[2: 163] = feature_data
        state_full = np.array(state, dtype=float)

        state_deep_copy = copy.deepcopy(state_full)
        state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget, state_deep_copy[1] / auc_num

        current_data_ctr = float(test_ctr[i])  # 当前数据的ctr，原始为str，应该转为float

        # 下一个状态的特征（除去预算、剩余拍卖数量）
        auc_data_next = test_data.iloc[i + 1: i + 2, :].values.flatten().tolist()[0: 16]
        next_feature_data = [test_ctr[i + 1]]
        for feat_next in auc_data_next:
            next_feature_data += embedding_v.iloc[feat_next, :].values.tolist()
        auc_data_next = np.array(next_feature_data, dtype=float).tolist()
        if current_data_ctr >= test_avg_ctr[int(hour_index)]:
            # RL代理根据状态选择动作
            action = RL.choose_best_action(state_deep_copy)

            # RL采用动作后获得下一个状态的信息以及奖励
            state_, reward, done, is_win = env.step_eCPI(auc_data, action, auc_data_next)
        else:
            action = 0
            state_, reward, done, is_win = env.step_eCPI(auc_data, action, auc_data_next)

        if is_win:
            hour_clks[int(hour_index)] += int(reward)
            total_reward_clks += reward
            total_imps += 1

        real_clks += int(auc_data[16])
        real_hour_clks[int(hour_index)] += int(auc_data[16])

        if done:
            if state_[0] < 0:
                spent = budget
            else:
                spent = budget - state_[0]
            cpm = spent / total_imps
            result_array.append([total_reward_clks, i, total_imps, budget, spent, cpm, real_clks])
            break

    print('\n测试集中: 出价数{}, 赢标数{}, 总点击数{}, 真实点击数{}, 预算{}, 总花费{}, CPM{}\n'.format(result_array[0][1], result_array[0][2],
                                                                    result_array[0][0], result_array[0][6], result_array[0][3],
                                                                    result_array[0][4], result_array[0][5]))
    result_df = pd.DataFrame(data=result_array, columns=['clks', 'bids', 'imps(wins)', 'budget', 'spent', 'cpm', 'real_clks'])
    result_df.to_csv('../../result/DDQN_result_' + str(budget_para) + '.txt')

    hour_clks_array = {'hour_clks': hour_clks, 'real_hour_clks': real_hour_clks}
    hour_clks_df = pd.DataFrame(hour_clks_array)
    hour_clks_df.to_csv('../../result/DDQN_hour_clks.csv')

if __name__ == '__main__':
    e_greedy = 0.9 # epsilon

    env = AD_env()
    RL = DoubleDQN([action for action in np.arange(0, 300)], # 按照数据集中的“块”计量
              env.action_numbers, env.feature_numbers,
              learning_rate=0.01, # DQN更新公式的学习率
              reward_decay=0.9, # 奖励折扣因子
              e_greedy=e_greedy, # 贪心算法ε
              replace_target_iter=2000, # 每200步替换一次target_net的参数
              memory_size=10000, # 经验池上限
              batch_size=1024, # 每次更新时从memory里面取多少数据出来，mini-batch
              # output_graph=True # 是否输出tensorboard文件
              )

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        train_budget, train_auc_numbers = config['train_budget']*budget_para[i], config['train_auc_num']
        test_budget, test_auc_numbers = config['test_budget']*budget_para[i], config['test_auc_num']
        run_env(train_budget, train_auc_numbers, e_greedy, budget_para[i])
        print('########测试结果########\n')
        test_env(test_budget, test_auc_numbers, budget_para[i])
    # RL.plot_cost() # 观看神经网络的误差曲线