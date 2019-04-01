from src.DRLB.reward_net import RewardNet
from src.DRLB.env import AD_env
import numpy as np
import pandas as pd
import copy
import datetime
from src.config import config


def run_env(budget, auc_num, budget_para):
    AD_env.build_env(budget, auc_num)  # 参数为训练集的(预算， 总展示次数)
    # 训练
    step = 0
    print('data loading\n')
    train_data = pd.read_csv("../../data/fm/train_fm.csv", header=None)
    train_data.iloc[:, config['data_hour_index']] = train_data.iloc[:, config['data_hour_index']].astype(
        int)  # 将时间序列设置为Int类型
    embedding_v = pd.read_csv("../../data/fm/embedding_v.csv", header=None)
    train_ctr = pd.read_csv("../../data/fm/train_ctr_pred.csv", header=None).drop(0, axis=0)  # 读取训练数据集中每条数据的pctr
    train_ctr.iloc[:, 1] = train_ctr.iloc[:, 1].astype(float)  # ctr为float类型
    train_ctr = train_ctr.iloc[:, 1].values
    train_avg_ctr = pd.read_csv("../../transform_precess/train_avg_ctrs.csv", header=None).iloc[:,
                    1].values  # 每个时段的平均点击率

    records_array = []  # 用于记录每一轮的最终奖励，以及赢标（展示的次数）
    cpc = 30000
    accumulate_reward = 0 # 累积奖励
    accumulate_spent = 0
    for episode in range(config['train_episodes']):
        # 初始化状态
        state = AD_env.reset(budget, auc_num)  # 参数为训练集的(预算， 总展示次数)
        # 此处的循环为训练数据的长度
        # 状态初始化为预算及拍卖数量，在循环内加上拍卖向量值

        # # 重置epsilon
        # RL.reset_epsilon(0.9)

        print('第{}轮'.format(episode + 1))
        hour_clks = [0 for i in range(0, 24)]  # 记录每个小时获得点击数
        real_hour_clks = [0 for i in range(0, 24)]  # 记录数据集中真实点击数

        total_reward_clks = 0
        total_imps = 0
        real_clks = 0  # 数据集真实点击数（到目前为止，或整个数据集）
        bid_nums = 0  # 出价次数
        real_imps = 0  # 真实曝光数

        ctr_action_records = []  # 记录模型出价以及真实出价，以及ctr（在有点击数的基础上）
        for i in range(auc_num):

            real_imps += 1

            # auction全部数据
            auc_data = train_data.iloc[i: i + 1, :].values.flatten().tolist()
            # auction所在小时段索引
            hour_index = auc_data[config['data_hour_index']]

            feature_data = [train_ctr[i] * 10]  # ctr特征，放大以便于加大其在特征中的地位
            # auction特征（除去click，payprice, hour）
            for feat in auc_data[0: config['data_feature_index']]:
                feature_data += embedding_v.iloc[feat, :].values.tolist()  # 获取对应特征的隐向量

            state[2: config['feature_num']] = feature_data
            state_full = np.array(state, dtype=float)
            # 预算以及剩余拍卖数量缩放，避免因预算及拍卖数量数值过大引起神经网络性能不好
            # 执行深拷贝，防止修改原始数据
            state_deep_copy = copy.deepcopy(state_full)
            state_deep_copy[0], state_deep_copy[1] = state_deep_copy[0] / budget, state_deep_copy[1] / auc_num

            current_data_ctr = float(train_ctr[i])  # 当前数据的ctr，原始为str，应该转为float

            # 出价次数
            bid_nums += 1
            total_imps += 1

            # RL代理从环境中选择动作
            action = auc_data[config['data_marketprice_index']]  # 1*17维,第三个参数为epsilon
            reward = cpc * current_data_ctr - action

            total_reward_clks += int(auc_data[config['data_clk_index']])
            real_clks += int(auc_data[config['data_clk_index']])
            accumulate_spent += action
            accumulate_reward += reward

            RewardNet.store_state_action_pair(state_deep_copy, action, accumulate_reward)

            # 当经验池数据达到一定量后再进行学习
            if step > config['batch_size']:
                RewardNet.learn()

            step += 1

            if bid_nums % 1000 == 0:
                if total_imps != 0:
                    now_cpm = accumulate_spent / total_imps
                else:
                    now_cpm = 0
                print('episode {}: 真实曝光数{}, 出价数{}, 赢标数{}, 当前点击数{}, 真实点击数{}, 预算{}, '
                      '花费{}, CPM{}\t{}'.format(episode, i, bid_nums, total_imps, total_reward_clks, real_clks,
                                               budget, accumulate_spent, now_cpm, datetime.datetime.now()))

        RewardNet.store_state_action_accumulate_reward()


if __name__ == '__main__':
    env = AD_env()
    RewardNet = RewardNet([action for action in np.arange(1, 301)],  # 按照数据集中的“块”计量
             1,153,)

    budget_para = config['budget_para']
    for i in range(len(budget_para)):
        train_budget, train_auc_numbers = config['train_budget'] * budget_para[i], config['train_auc_num']
        test_budget, test_auc_numbers = config['test_budget'] * budget_para[i], config['test_auc_num']
        run_env(train_budget, train_auc_numbers, budget_para[i])