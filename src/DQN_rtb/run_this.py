from env import AD_env
from RL_brain import DQN
import csv
import numpy as np
import pandas as pd

def run_env():
    env.build_env(22067108, 328481) # 参数为训练集的(预算， 总展示次数)
    # 训练
    step = 0
    print('data loading\n')
    train_data = pd.read_csv("../../data/normalized_train_data.csv", header=None)

    for episode in range(300):
        # 初始化状态
        state = env.reset(22067108/64, 328481) # 参数为训练集的(预算， 总展示次数)
        # 此处的循环为训练数据的长度
        # 状态初始化为预算及拍卖数量，在循环内加上拍卖向量值
        # 拍卖向量的特征考虑一下

        # 矩阵形式

        # print('第{}轮'.format(episode+1))
        # total_reward = 0
        # for i in range(len(train_data)):
        #
        #     random_index = np.random.randint(0, len(train_data))
        #
        #     # auction特征（除去click，payprice）
        #     feature_data = train_data.iloc[random_index: random_index + 1, 0:15].values.flatten().tolist()
        #     #print(data.iloc[:, 0:15]) # 取前10列的数据，逗号前面的冒号表示取所有行，逗号后面的冒号表示取得列的范围，如果只有一个貌似就表示取所有列，行同理
        #     state[2: 17] = feature_data
        #     state_full = np.array(state)
        #     # RL代理根据状态选择动作
        #
        #     action = RL.choose_action(state_full) # 1*17维
        #
        #     # auction全部数据
        #     data = train_data.iloc[random_index: random_index + 1, :].values.flatten().tolist()
        print('第{}轮'.format(episode + 1))
        total_reward = 0
        for i in range(len(train_data)):

            # auction特征（除去click，payprice）
            feature_data = train_data.iloc[i: i + 1, 0:15].values.flatten().tolist()
            # print(data.iloc[:, 0:15]) # 取前10列的数据，逗号前面的冒号表示取所有行，逗号后面的冒号表示取得列的范围，如果只有一个貌似就表示取所有列，行同理
            state[2: 17] = feature_data
            state_full = np.array(state)
            # RL代理根据状态选择动作

            action = RL.choose_action(state_full)  # 1*17维
            # auction全部数据
            data = train_data.iloc[i: i + 1, :].values.flatten().tolist()
            # RL采用动作后获得下一个状态的信息以及奖励
            state_, reward, done = env.step(data, action) # 这里的data明天看一下

            total_reward += reward
            # RL代理将 状态-动作-奖励-下一状态 存入经验池
            RL.store_transition(state, action, reward, state_)

            # 当经验池数据达到一定量后再进行学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 将下一个state_变为 下次循环的state
            state = state_

            # 如果终止（满足一些条件），则跳出循环
            if done:
                print(i)
                break
            step += 1
        print('第{}轮总奖励{}'.format(episode, total_reward))
        print('训练结束\n')


def test_env():
    env.build_env(14560732/64, 191335) # 参数为测试集的(预算， 总展示次数)
    state = env.reset(14560732/64, 191335) # 参数为测试集的(预算， 总展示次数)


    train_data = pd.read_csv("../../data/normalized_test_data.csv", header=None)

    total_reward = 0
    for i in range(len(train_data)):
        if i == 0:
            continue
        # 刷新环境env
        # env.render()

        # 二维矩阵转一维，用flatten函数
        # auction特征（除去click，payprice）
        feature_data = train_data.iloc[i: i+1, 0:15].values.flatten().tolist()
        state[2: 17] = feature_data

        state_full = np.array(state)
        # RL代理根据状态选择动作

        action = RL.choose_action(state_full)

        # auction全部数据
        data = train_data.iloc[i: i + 1, :].values.flatten().tolist()
        # RL采用动作后获得下一个状态的信息以及奖励
        state_, reward, done = env.step(data, action)

        total_reward += reward

        if done:
            break

    print('总收益为{}'.format(total_reward))


if __name__ == '__main__':
    env = AD_env()
    RL = DQN([action for action in np.arange(0, 300, 0.01)],
              env.action_numbers, env.feature_numbers,
              learning_rate=0.01, # DQN更新公式的学习率
              reward_decay=0.9, # 奖励折扣因子
              e_greedy=0.9, # 贪心算法ε
              replace_target_iter=200, # 每200步替换一次target_net的参数
              memory_size=20000, # 经验池上限
              # output_graph=True # 是否输出tensorboard文件
              )
    run_env()
    test_env()
    RL.plot_cost() # 观看神经网络的误差曲线