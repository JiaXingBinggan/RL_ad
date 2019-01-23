import sys
import random
import math
import pandas as pd
import numpy as np
random.seed(10)

def bidding_const(bid):
    return bid

def bidding_rand(upper):
    return int(random.random() * upper)

def bidding_mcpc(ecpc, pctr):
    return int(ecpc * pctr)

def bidding_lin(pctr, base_ctr, base_bid):
    return int(pctr * base_bid / base_ctr)

def bidding_opt(pCTR, lamda=5.2*1e-7):  # 出价策略函数
    c = 20
    temp1 = (pCTR + math.sqrt((c * lamda) ** 2 + pCTR ** 2) / c * lamda)
    temp2 = c * lamda / (pCTR + math.sqrt((c * lamda) ** 2 + pCTR ** 2))
    bid_price = c * (temp1 ** (1 / 3) - temp2 ** (1 / 3))
    return bid_price

def win_auction(case, bid):
    return bid > case[1] # bid > winning price

# 从训练数据中读取到初始ecpc和初始ctr
train_data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None).drop(0, axis=0)
train_data.values[:, [0, 23]] = train_data.values[:, [0, 23]].astype(int)
imp_num = len(train_data.values)
original_ctr = np.sum(train_data.values[:, 0]) / imp_num
original_ecpc = np.sum(train_data.values[:, 23]) / np.sum(train_data.values[:, 0])

clicks_prices = [] # clk and price
total_cost = 0 # total original cost during the test data
# 从测试数据中读取测试数据
test_data = pd.read_csv('../../sample/20130613_test_sample.csv', header=None).drop(0, axis=0)
test_data.values[:, [0, 23]] = test_data.values[:, [0, 23]].astype(int)
data = test_data.values
for i in range(len(data)):
    click = int(data[i][0])
    winning_price = int(data[i][23])
    clicks_prices.append((click, winning_price, int(data[i][2])))

total_cost += test_data.iloc[:, 23].sum()

print('总预算{}'.format(total_cost))
# budgetProportion clk cnv bid imp budget spend para
def simulate_one_bidding_strategy_with_parameter(cases, ctrs, tcost, proportion, algo, para):
    budget = int(tcost / proportion) # intialise the budget
    cpc = 30000 # cost per click
    revenue = 350 # 收益

    cost = 0
    clks = 0
    bids = 0
    imps = 0
    profits = 0

    real_imps = 0
    real_clks = 0
    for idx in range(0, len(cases)):
        bid = 0
        pctr = ctrs[idx]
        if algo == "const":
            bid = bidding_const(para)
        elif algo == "rand":
            bid = bidding_rand(para)
        elif algo == "mcpc":
            bid = bidding_mcpc(original_ecpc, pctr)
        elif algo == "lin":
            bid = bidding_lin(pctr, original_ctr, para)
        elif algo == "bidding_opt":
            bid = bidding_opt(pctr)
        else:
            print('wrong bidding strategy name')
            sys.exit(-1)
        bids += 1
        case = cases[idx]
        real_imps += 1
        real_clks += case[0]
        if win_auction(case, bid):
            imps += 1
            clks += case[0]
            cost += case[1]
            profits += (revenue*case[0] - case[1])
        if cost > budget:
            print('早停时刻', case[2])
            break
    cpm = 0
    cpm = (cost / imps) if imps > 0 else 0
    return str(proportion) + '\t' + str(profits) + '\t' + str(clks) + '\t' + str(real_clks) + '\t' + str(bids) + '\t' + \
        str(imps) + '\t' + str(real_imps) + '\t' + str(budget) + '\t' + str(cost) + '\t' + str(cpm) + '\t'+ algo + '\t' + str(para)

def simulate_one_bidding_strategy(cases, ctrs, tcost, proportion, algo, writer):
    paras = algo_paras[algo]
    for para in paras:
        res = simulate_one_bidding_strategy_with_parameter(cases, ctrs, tcost, proportion, algo, para)
        print(res)
        writer.write(res + '\n')

pctrs = []
# ctr_fi = open('../../data/test_lr_pred.csv', 'r')
# for line in ctr_fi:
#     pctrs.append(float(line.split('.')[1].strip()))
#     print(float(line.split('.')[1].strip()))
# ctr_fi.close()
test_ctrs = pd.read_csv('../../data/fm/test_ctr_pred.csv', header=None).drop(0, axis=0)
test_ctrs.iloc[:, 1] = test_ctrs.iloc[:, 1].astype(float)
pctrs = test_ctrs.iloc[:, 1].values.flatten().tolist()

# parameters setting for each bidding strategy
budget_proportions = [64]
const_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10))
rand_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) +list(np.arange(100, 301, 10))
mcpc_paras = [1]
lin_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10))

algo_paras = {"const":const_paras, "rand":rand_paras, "mcpc":mcpc_paras, "lin":lin_paras, "bidding_opt": [0]}

fo = open('../../result/results.txt', 'w') # rtb.results.txt
header = "prop\tprofits\tclks\treal_clks\tbids\timps\treal_imps\tbudget\tspend\tcpm\talgo\tpara"
fo.write(header + '\n')
print(header)
for proportion in budget_proportions:
    for algo in algo_paras:
        simulate_one_bidding_strategy(clicks_prices, pctrs, total_cost, proportion, algo, fo)

