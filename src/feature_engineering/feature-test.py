from sklearn.datasets import load_iris
import sklearn.preprocessing as ppr
import sklearn.feature_selection as fea_s
'''
iris 数据集
Classes 	3
Samples per class 	50
Samples total 	150
Dimensionality 	4
Features 	real, positive
'''
# 导入IRIS数据集
iris = load_iris()

# IRIS特征矩阵
print(iris.data)

# IRIS目标向量
# print(iris.target)

'''
数据预处理
'''
# 标准化，返回值为标准化后的数据
stand_data = ppr.StandardScaler().fit_transform(iris.data)
# print(stand_data)

# 区间缩放法，返回为缩放到[0, 1]区间的数据
min_max_data = ppr.MinMaxScaler().fit_transform(iris.data)
# print(min_max_data)

# 使用preproccessing库的Normalizer类对数据进行归一化的代码
normalized_data = ppr.Normalizer().fit_transform(iris.data)
# print(normalized_data)

# 二值化，阈值设置为3，返回值为二值化后的数据
binarized_data = ppr.Binarizer(threshold=3).fit_transform(iris.data)
# print(binarized_data)

# 哑编码
one_hot_data = ppr.OneHotEncoder().fit_transform(iris.data)
# print(one_hot_data)

# 缺失值计算

# 数据变换

'''
特征选择
'''
# 　使用方差选择法，先要计算各个特征的方差，然后根据阈值，
#  选择方差大于阈值的特征。使用feature_selection库的VarianceThreshold类来选择特征的代码如下
variance_select = fea_s.VarianceThreshold(threshold=3).fit_transform(iris.data)
print(variance_select) # 返回的方差大于3的那一列特征值


