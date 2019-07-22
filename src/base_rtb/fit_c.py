import numpy as np
from scipy.optimize import curve_fit

def win_rate_f(bid, c):
    win_rate = bid / (c + bid)
    return win_rate

def fit_c(train_data):
    win_rate_dict = {}
    train_data_length = len(train_data)
    for price in range(301):
        price_win_nums = np.sum(price >= train_data.iloc[:, 23])
        win_rate_dict[price] = price_win_nums / train_data_length

    x_data = np.arange(0, 301)
    y_data = list(win_rate_dict.values())
    popt, pcov = curve_fit(win_rate_f, x_data, y_data)

    return popt[0]
