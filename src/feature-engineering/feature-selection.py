import sklearn.feature_selection as f_s
import pandas as pd

train_data = pd.read_csv("../../data/train_encoder_data.csv", header=None)
# feature_data = train_data.drop([0, 23], axis=1)
variance_array = f_s.VarianceThreshold(threshold=3)
variance_array.fit(train_data.values)

# 打印各个特征的方差
print(variance_array.variances_)

# 打印方差大于3的特征数组
print(variance_array.transform(train_data.values))

print(variance_array.transform(train_data.values)[0])

# 打印方差大于3的特征下标
print(variance_array.get_support(True))
# [ 1  2  3  4  6  7  8  9 10 12 13 15 16 20 21 23 26]
# 除去0号特征是否点击，3号特征时间戳timestamp，第23号特征为payprice，因此选择特征为
# # [ 1  2  4  6  7  8  9 10 12 13 15 16 20 21 26]

train_data = pd.read_csv("../../data/test_encoder_data.csv", header=None)
# feature_data = train_data.drop([0, 23], axis=1)
variance_array = f_s.VarianceThreshold(threshold=3)
variance_array.fit(train_data.values)

# 打印各个特征的方差
print(variance_array.variances_)

# 打印方差大于3的特征数组
print(variance_array.transform(train_data.values))

print(variance_array.transform(train_data.values)[0])

# 打印方差大于3的特征下标
print(variance_array.get_support(True))
# [ 1  2  3  4  6  7  8  9 10 12 13 15 16 20 21 23 26]
# 除去0号特征是否点击，3号特征时间戳timestamp，第23号特征为payprice，因此选择特征为
# [ 1  2  4  6  7  8  9 10 12 13 15 16 20 21 26]