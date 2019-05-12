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

def bidding_opt(pCTR, lamda=5.2e-7):  # 出价策略函数
    c = 46.98063452
    # temp1 = (pCTR + math.sqrt((c * lamda) ** 2 + pCTR ** 2) / c * lamda)
    # temp2 = c * lamda / (pCTR + math.sqrt((c * lamda) ** 2 + pCTR ** 2))
    # bid_price = c * (temp1 ** (1 / 3) - temp2 ** (1 / 3))
    bid_price = math.sqrt(c*pCTR/lamda + c**2) -c
    return bid_price

def win_auction(case, bid):
    return bid >= case[1] # bid > winning price

# 从训练数据中读取到初始ecpc和初始ctr
# train_data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None).drop(0, axis=0)
train_data = pd.read_csv('../../data/20130606_train_data.csv', header=None).drop(0, axis=0)
train_data.values[:, [0, 23]] = train_data.values[:, [0, 23]].astype(int)
imp_num = len(train_data.values)
original_ctr = np.sum(train_data.values[:, 0]) / imp_num
original_ecpc = np.sum(train_data.values[:, 23]) / np.sum(train_data.values[:, 0])

clicks_prices = [] # clk and price
total_cost = 0 # total original cost during the test data
# 从测试数据中读取测试数据
test_data = pd.read_csv('../../sample/20130607_test_data.csv', header=None).drop(0, axis=0)
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

    cost = 0
    clks = 0
    bids = 0
    imps = 0
    profits = 0

    real_imps = 0
    real_clks = 0
    for idx in range(0, len(cases)):
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
            # print(bid)
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
            profits += (cpc*pctr - case[1])
        if cost > budget:
            print('早停时刻', case[2])
            break
    cpm = (cost / imps) if imps > 0 else 0
    return str(proportion) + '\t' + str(profits) + '\t' + str(clks) + '\t' + str(real_clks) + '\t' + str(bids) + '\t' + \
        str(imps) + '\t' + str(real_imps) + '\t' + str(budget) + '\t' + str(cost) + '\t' + str(cpm) + '\t'+ algo + '\t' + str(para)

def simulate_one_bidding_strategy(cases, ctrs, tcost, proportion, algo, writer):
    paras = algo_paras[algo]
    for para in paras:
        res = simulate_one_bidding_strategy_with_parameter(cases, ctrs, tcost, proportion, algo, para)
        print(res)
        writer.write(res + '\n')

test_ctrs = pd.read_csv('../../data/fm/test_ctr_pred.csv', header=None).drop(0, axis=0)
test_ctrs.iloc[:, 1] = test_ctrs.iloc[:, 1].astype(float)
pctrs = test_ctrs.iloc[:, 1].values.flatten().tolist()

# parameters setting for each bidding strategy
budget_proportions = [1, 2, 4, 8, 16]
algos = ["const", "rand", "mcpc", "lin"]
const_paras = {}
rand_paras = {}
lin_paras = {}
result_best_fi = open('../../result/results_train.best.perf.txt', 'r')
for i, line in enumerate(result_best_fi):
    if i == 0:
        continue
    for budget_proportion in budget_proportions:
        if budget_proportion == int(line.split('\t')[0]):
            for algo in algos:
                current_algo = line.split('\t')[10]
                current_para = int(line.split('\t')[11])
                if current_algo == 'const':
                    const_paras[budget_proportion] = current_para
                elif current_algo == 'rand':
                    rand_paras[budget_proportion] = current_para
                elif current_algo == 'lin':
                    lin_paras[budget_proportion] = current_para
const_paras = [item[1] for item in sorted(const_paras.items(),key=lambda d:d[0],reverse=False)]
rand_paras = [item[1] for item in sorted(rand_paras.items(),key=lambda d:d[0],reverse=False)]
lin_paras = [item[1] for item in sorted(lin_paras.items(),key=lambda d:d[0],reverse=False)]

print(const_paras, rand_paras, lin_paras)
fo = open('../../result/results_test.txt', 'w') # rtb.results.txt
header = "prop\tprofits\tclks\treal_clks\tbids\timps\treal_imps\tbudget\tspend\tcpm\talgo\tpara"
fo.write(header + '\n')
print(header)
for k, proportion in enumerate(budget_proportions):
    algo_paras = {"const": [const_paras[k]], "rand": [rand_paras[k]], "mcpc": [1], "lin": [lin_paras[k]], "bidding_opt": [0]}
    for algo in algo_paras:
        simulate_one_bidding_strategy(clicks_prices, pctrs, total_cost, proportion, algo, fo)

