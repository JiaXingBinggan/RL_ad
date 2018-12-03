from env import AD_env
from RL_brain import DoubleDQN
import csv
import numpy as np
import pandas as pd


def run_env(budget, auc_num, e_greedy):
    env.build_env(budget, auc_num) # 参数为训练集的(预算， 总展示次数)
    # 训练
    step = 0
    print('data loading\n')
    train_data = pd.read_csv("../../data/normalized_train_data.csv", header=None)
    train_lr = pd.read_csv("../../data/train_lr_pred.csv", header=None).iloc[:, 1].values # 读取训练数据集中每条数据的pctr
    train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:,1].values # 每个时段的平均点击率

    records_array = [] # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    for episode in range(300):
        # 初始化状态
        state = env.reset(budget, auc_num) # 参数为训练集的(预算， 总展示次数)
        # 此处的循环为训练数据的长度
        # 状态初始化为预算及拍卖数量，在循环内加上拍卖向量值

        # 重置epsilon
        RL.reset_epsilon(0.9)

        print('第{}轮'.format(episode + 1))
        total_reward = 0
        for i in range(len(train_data)):
            # auction全部数据
            random_index = np.random.randint(0, len(train_data))
            auc_data = train_data.iloc[random_index: random_index + 1, :].values.flatten().tolist()
            # auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()

            # auction所在小时段索引
            hour_index = auc_data[17]

            # auction特征（除去click，payprice, hour）
            feature_data = auc_data[0:15]
            # print(data.iloc[:, 0:15]) # 取前10列的数据，逗号前面的冒号表示取所有行，逗号后面的冒号表示取得列的范围，如果只有一个貌似就表示取所有列，行同理
            state[2: 17] = feature_data
            state_full = np.array(state)

            if train_lr[random_index] >= train_avg_ctr[int(hour_index)]:
                # RL代理根据状态选择动作
                action = RL.choose_action(state_full, train_lr[random_index], e_greedy)  # 1*17维,第三个参数为epsilon

                # RL采用动作后获得下一个状态的信息以及奖励
                state_, reward, done = env.step(auc_data, action)
                # RL代理将 状态-动作-奖励-下一状态 存入经验池
            else:
                action = 0 # 出价为0，即不参与竞标
                state_, reward, done = env.step(auc_data, action)

            RL.store_transition(state, action, reward, state_)
            total_reward += reward

            # 当经验池数据达到一定量后再进行学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个state_变为 下次循环的state
            state = state_

            # 如果终止（满足一些条件），则跳出循环
            if done:
                print(i)
                records_array.append([total_reward, i])
                break
            step += 1
        print('第{}轮总奖励{}'.format(episode, total_reward))
        print('训练结束\n')

        records_df = pd.DataFrame(data=records_array, columns=['clks', 'imps'])
        records_df.to_csv('../results/DQN_train.txt')


def test_env(budget, auc_num, e_greedy):
    env.build_env(budget, auc_num) # 参数为测试集的(预算， 总展示次数)
    state = env.reset(budget, auc_num) # 参数为测试集的(预算， 总展示次数)

    test_data = pd.read_csv("../../data/normalized_test_data.csv", header=None)
    test_lr = pd.read_csv("../../data/test_lr_pred.csv", header=None).iloc[:, 1].values  # 读取测试数据集中每条数据的pctr
    test_avg_ctr = pd.read_csv("../../transform_precess/test_avg_ctrs.csv", header=None).iloc[:,1].values  # 测试集中每个时段的平均点击率

    result_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）

    total_reward = 0
    for i in range(len(test_data)):
        if i == 0:
            continue
        # auction全部数据
        auc_data = test_data.iloc[i: i + 1, :].values.flatten().tolist()

        # auction所在小时段索引
        hour_index = auc_data[17]

        # 二维矩阵转一维，用flatten函数
        # auction特征（除去click，payprice）
        feature_data = auc_data[0:15]
        state[2: 17] = feature_data
        state_full = np.array(state)

        if test_lr[i] >= test_avg_ctr[int(hour_index)]:
            # RL代理根据状态选择动作
            action = RL.choose_best_action(state_full)

            # RL采用动作后获得下一个状态的信息以及奖励
            state_, reward, done = env.step(auc_data, action)
        else:
            action = 0
            state_, reward, done = env.step(auc_data,action)

        total_reward += reward

        if done:
            result_array.append([total_reward, i])
            break

    print('总收益为{}'.format(total_reward))

    result_df = pd.DataFrame(data=result_array, columns=['clks', 'imps'])
    result_df.to_csv('../results/DQN_train.txt')


if __name__ == '__main__':
    e_greedy = 0.9 # epsilon

    env = AD_env()
    RL = DoubleDQN([action for action in np.arange(0, 301, 0.01)],
              env.action_numbers, env.feature_numbers,
              learning_rate=0.01, # DQN更新公式的学习率
              reward_decay=0.9, # 奖励折扣因子
              e_greedy=e_greedy, # 贪心算法ε
              replace_target_iter=200, # 每200步替换一次target_net的参数
              memory_size=2000, # 经验池上限
              batch_size=128, # 每次更新时从memory里面取多少数据出来，mini-batch
              # output_graph=True # 是否输出tensorboard文件
              )
    train_budget, train_auc_numbers = 22067108/64, 328481
    test_budget, test_auc_numbers = 14560732/64, 191335
    run_env(train_budget, train_auc_numbers, e_greedy)
    test_env(test_budget, test_auc_numbers, e_greedy)
    # RL.plot_cost() # 观看神经网络的误差曲线