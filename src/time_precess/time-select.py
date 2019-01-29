# 用于统计不同时段流量
import pandas as pd
import numpy as np

# 训练集下
data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None).drop(0, axis=0)

day_array = []
hour_array = []

date_data = data.iloc[:, 4].values
for i in range(0, len(date_data)):
    str_data = str(date_data[i])

    day_data = str_data[0:8]
    hour_data = str_data[8:10]

    if day_data not in day_array:
        day_array.append(day_data)
    if hour_data not in hour_array:
        hour_array.append(hour_data)


day = {'day': day_array, 'clks': 0}
day_df = pd.DataFrame(data=day)

# 两种方式建立多层索引
# 1.pivot_tabel方法，需要对原数据进行处理
# hour = {'day': day_array*24, 'hour': hour_array*7, 'clks': 0}
# hour_df = pd.DataFrame(data=hour)
# hour_table = pd.pivot_table(hour_df, index=['day', 'hour'])
# 2.product方法，不需要处理原数据
hour = {'clks': 0}
hour_df = pd.DataFrame(data=hour, index=pd.MultiIndex.from_product([day_array, hour_array])).unstack()

click_data = data.iloc[:, 0 : 4].values
day_df_index = day_df.values
hour_df_index = hour_df.values
for k in range(0, len(date_data)):
    str_data = str(date_data[k])

    day_data = str_data[0:8]
    hour_data = str_data[8:10]

    # 匹配数据集中时间（天）和pandas dataframe中的时间（天），返回下标
    day_compare_index = int(np.argwhere(day_df_index[:, 0] == day_data))
    day_df_index[day_compare_index, 1] += int(click_data[k, 0]) # 还没有赋值给对应的字段

    hour_compare_index = int(np.argwhere(np.array(hour_array) == hour_data))
    hour_df_index[day_compare_index, hour_compare_index] += int(click_data[k, 0])

day_df.iloc[:,1] = day_df_index[:, 1]
day_df.to_csv('../../transform_precess/day_select_result.csv', header=None, index=None)

hour_df.iloc[:, :] = hour_df_index[:, :]
hour_df.to_csv('../../transform_precess/hour_select_result.csv', header=None, index=None)

# 测试集下
data = pd.read_csv('../../sample/20130612_test_sample.csv', header=None).drop(0, axis=0)

day_array = []
hour_array = []

date_data = data.iloc[:, 4].values
for i in range(0, len(date_data)):
    str_data = str(date_data[i])

    day_data = str_data[0:8]
    hour_data = str_data[8:10]

    if day_data not in day_array:
        day_array.append(day_data)
    if hour_data not in hour_array:
        hour_array.append(hour_data)


day = {'day': day_array, 'clks': 0}
day_df = pd.DataFrame(data=day)

# 两种方式建立多层索引
# 1.pivot_tabel方法，需要对原数据进行处理
# hour = {'day': day_array*24, 'hour': hour_array*7, 'clks': 0}
# hour_df = pd.DataFrame(data=hour)
# hour_table = pd.pivot_table(hour_df, index=['day', 'hour'])
# 2.product方法，不需要处理原数据
hour = {'clks': 0}
hour_df = pd.DataFrame(data=hour, index=pd.MultiIndex.from_product([day_array, hour_array])).unstack()

click_data = data.iloc[:, 0 : 4].values
day_df_index = day_df.values
hour_df_index = hour_df.values
for k in range(0, len(date_data)):
    str_data = str(date_data[k])

    day_data = str_data[0:8]
    hour_data = str_data[8:10]

    # 匹配数据集中时间（天）和pandas dataframe中的时间（天），返回下标
    day_compare_index = int(np.argwhere(day_df_index[:, 0] == day_data))
    day_df_index[day_compare_index, 1] += int(click_data[k, 0]) # 还没有赋值给对应的字段

    hour_compare_index = int(np.argwhere(np.array(hour_array) == hour_data))
    hour_df_index[day_compare_index, hour_compare_index] += int(click_data[k, 0])

day_df.iloc[:,1] = day_df_index[:, 1]
day_df.to_csv('../../transform_precess/test_day_select_result.csv', header=None, index=None)

hour_df.iloc[:, :] = hour_df_index[:, :]
hour_df.to_csv('../../transform_precess/test_hour_select_result.csv', header=None, index=None)
