import pandas as pd
import sklearn.preprocessing as ppr
import numpy as np

# 处理缺失值
# 训练数据
train_missing_data = pd.read_csv("../../sample/20130606_train_sample.csv", header=None)
fill_data = train_missing_data.fillna(value='other')
fill_data.to_csv('../sample/../train_sample_data.csv', index=False, header=False)
print('训练数据特征缺失值处理完成\n')

# 训练数据特征编码LabelEncoder（也可用OneHotEncoder）
train_data = pd.read_csv('../../sample/train_sample_data.csv', header=None)
train_encoder_df = pd.DataFrame({})
for i in range(0, 27):
    if i != 0 or i !=23: # click 和 payprice不做编码
        # 也可以用OneHotEncoder
        label_data = ppr.LabelEncoder().fit_transform(train_data.iloc[:, i:i+1].values.reshape(-1, 1))
        train_encoder_df[i] = label_data
    else:
        train_encoder_df[i] = train_data.iloc[:, i:i+1].values.reshape(-1, 1)
train_encoder_df.to_csv('../../data/train_encoder_data.csv', index=False, header=False)
print('训练数据特征编码完成\n')

# 训练数据
test_missing_data = pd.read_csv("../../sample/20130613_test_sample.csv", header=None)
fill_data = test_missing_data.fillna(value='other')
fill_data.to_csv('../../sample/test_sample_data.csv', index=False, header=False)
print('测试数据特征缺失值处理完成\n')

# # 特征编码LabelEncoder（也可用OneHotEncoder）
test_data = pd.read_csv('../../sample/test_sample_data.csv', header=None)
test_encoder_df = pd.DataFrame({})
for i in range(0, 29):
    if i != 0 or i !=23: # click 和 payprice不做编码
        # 也可以用OneHotEncoder
        label_data = ppr.LabelEncoder().fit_transform(test_data.iloc[:, i:i+1].values.reshape(-1, 1))
        test_encoder_df[i] = label_data
    else:
        test_encoder_df[i] = test_data.iloc[:, i:i+1].values.reshape(-1, 1)
test_encoder_df.to_csv('../../data/test_encoder_data.csv', index=False, header=False)
print('测试数据特征编码完成\n')