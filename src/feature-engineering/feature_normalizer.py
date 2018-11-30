import numpy as np
from sklearn import preprocessing
import csv
import pandas as pd

train_data = pd.read_csv("../../data/train_encoder_data.csv", header=None)
# 选择[ 1  2  4  6  7  8  9 10 12 13 15 16 20 21 26]号特征
data = train_data.drop([0, 3, 5, 11, 14, 17, 18, 19, 22, 23, 24, 25], axis=1) # 去除这些特征

min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(data)

data =pd.DataFrame(X_minMax)

data['clk'] = train_data[0]
data['payprice'] = train_data[23]
data['hour'] = train_data[2]
print(data.head())

data.to_csv('../../data/normalized_train_data.csv', index=False, header=False)

test_data = pd.read_csv("../../data/test_encoder_data.csv", header=None)
# 选择[ 1  2  4  6  7  8  9 10 12 13 15 16 20 21 26]号特征
data = test_data.drop([0, 3, 5, 11, 14, 17, 18, 19, 22, 23, 24, 25, 27, 28], axis=1) # 去除这些特征，同训练数据保持一致

min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(data)

data =pd.DataFrame(X_minMax)

data['clk'] = test_data[0]
data['payprice'] = test_data[23]
data['hour'] = train_data[2]
print(data.head())

data.to_csv('../../data/normalized_test_data.csv', index=False, header=False)

