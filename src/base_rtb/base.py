import sys
import random
import math
import pandas as pd
import numpy as np
from src.base_rtb.fit_c import fit_c
import os
random.seed(10)

def bidding_const(bid):
    return bid

def bidding_rand(upper):
    return int(random.random() * upper)

def bidding_mcpc(ecpc, pctr):
    return int(ecpc * pctr)

def bidding_lin(pctr, base_ctr, base_bid):
    return int(pctr * base_bid / base_ctr)

def bidding_opt(c, pCTR, lamda=5.2e-7):  # 出价策略函数
    bid_price = math.sqrt(c*pCTR/lamda + c**2) -c
    return bid_price

def win_auction(case, bid):
    return bid >= case[1] # bid > winning price

# budgetProportion clk cnv bid imp budget spend para
def simulate_one_bidding_strategy_with_parameter(data, bidding_opt_c, cases, ctrs, tcost, proportion, algo, para):
    budget = 10000000 # intialise the budget
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
            bid = bidding_opt(bidding_opt_c, pctr)
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

def simulate_one_bidding_strategy(data, bidding_opt_c, cases, ctrs, tcost, proportion, algo, writer):
    paras = algo_paras[algo]
    for para in paras:
        res = simulate_one_bidding_strategy_with_parameter(data, bidding_opt_c, cases, ctrs, tcost, proportion, algo, para)
        print(res)
        writer.write(res + '\n')

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.mkdir('result')

    campaign = '1458'
    # 从训练数据中读取到初始ecpc和初始ctr
    train_data = pd.read_csv('data/' + campaign + '/train_data.csv', header=None).drop(0, axis=0)
    train_data.iloc[:, 1: 4] \
        = train_data.iloc[:, 1 : 4].astype(
        int)
    train_data.iloc[:, 4] \
        = train_data.iloc[:, 4].astype(
        float)
    imp_num = len(train_data.values)
    original_ctr = np.sum(train_data.values[:, 1]) / imp_num
    original_ecpc = np.sum(train_data.values[:, 2]) / np.sum(train_data.values[:, 1])
    print(original_ctr)

    bidding_opt_c = fit_c(train_data)

    clicks_prices = [] # clk and price
    total_cost = 0 # total original cost during the train data
    data = train_data.values
    for i in range(len(data)):
        click = int(data[i][1])
        winning_price = int(data[i][2])
        clicks_prices.append((click, winning_price, int(data[i][3])))
    total_cost += train_data.iloc[:, 2].sum()

    print('总预算{}'.format(total_cost))

    # train_ctrs = pd.read_csv('../../data/fm/train_ctr_pred.csv', header=None).drop(0, axis=0)
    # train_ctrs.iloc[:, 1] = train_ctrs.iloc[:, 1].astype(float)
    pctrs = train_data.values[:, 4].flatten().tolist()

    # parameters setting for each bidding strategy
    budget_proportions = [1]
    const_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10))
    rand_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) +list(np.arange(100, 301, 10))
    mcpc_paras = [1]
    lin_paras = list(np.arange(2, 20, 2)) + list(np.arange(20, 100, 5)) + list(np.arange(100, 301, 10))

    # algo_paras = {"const":const_paras, "rand":rand_paras, "mcpc": mcpc_paras, "lin": lin_paras, "bidding_opt": [0]}
    algo_paras = {"lin": lin_paras}

    fo = open('result/' + campaign + '/results_train.txt', 'w') # rtb.results.txt
    header = "prop\tprofits\tclks\treal_clks\tbids\timps\treal_imps\tbudget\tspend\tcpm\talgo\tpara"
    fo.write(header + '\n')
    print(header)
    for proportion in budget_proportions:
        for algo in algo_paras:
            simulate_one_bidding_strategy(train_data, bidding_opt_c, clicks_prices, pctrs, total_cost, proportion, algo, fo)

