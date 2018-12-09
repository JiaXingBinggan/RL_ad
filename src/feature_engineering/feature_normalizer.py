import numpy as np
from sklearn import preprocessing
import csv
import pandas as pd

train_data = pd.read_csv("../../data/train_encoder_data.csv", header=None)
# 选择[1, 2, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 25]号特征
data = train_data.drop([0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26], axis=1) # 去除这些特征
# [0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26]

min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(data)

data =pd.DataFrame(X_minMax)

data['clk'] = train_data[0]
data['payprice'] = train_data[23]
data['hour'] = train_data[2]
print(data.head())

data.to_csv('../../data/normalized_train_data.csv', index=False, header=False)

test_data = pd.read_csv("../../data/test_encoder_data.csv", header=None)
# 选择[1, 2, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 25]号特征
data = test_data.drop([0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26, 27, 28], axis=1) # 去除这些特征，同训练数据保持一致
# [0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26, 27, 28]

min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(data)

data =pd.DataFrame(X_minMax)

data['clk'] = test_data[0]
data['payprice'] = test_data[23]
data['hour'] = train_data[2]
print(data.head())

data.to_csv('../../data/normalized_test_data.csv', index=False, header=False)

train_data = pd.read_csv("../../data/train_encoder_data.csv", header=None)
# 选择[1, 2, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 25]号特征
data = train_data.drop([0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26], axis=1) # 去除这些特征
# [0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26]
data['clk'] = train_data[0]
data['payprice'] = train_data[23]
data['hour'] = train_data[2]
print(data.head())
data.to_csv('../../data/selected_train_data.csv', index=False, header=False)

test_data = pd.read_csv("../../data/test_encoder_data.csv", header=None)
# 选择[1, 2, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 25]号特征
data = test_data.drop([0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26, 27, 28], axis=1) # 去除这些特征，同训练数据保持一致
# [0, 3, 4, 5, 6, 13, 14, 22, 23, 24, 26, 27, 28]
data['clk'] = test_data[0]
data['payprice'] = test_data[23]
data['hour'] = train_data[2]
print(data.head())

data.to_csv('../../data/seleted_test_data.csv', index=False, header=False)

