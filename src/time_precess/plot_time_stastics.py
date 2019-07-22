import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

day_select_result = pd.read_csv('../../transform_precess/day_select_result.csv').values
hour_select_result = pd.read_csv('../../transform_precess/hour_select_result.csv').values

x_axis = np.arange(0,24)
y_aixs_1 = hour_select_result[0]
y_aixs_2 = hour_select_result[1]

plt.plot(x_axis, y_aixs_1, 'r', label='day1')
plt.plot(x_axis, y_aixs_2, 'b', label='day2')
plt.legend()
plt.show()