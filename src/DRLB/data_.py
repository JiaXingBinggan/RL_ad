import pandas as pd
import numpy as np

train_data = pd.read_csv('../../sample/20130606_train_sample.csv', header=None).drop([0])
train_ctrs = pd.read_csv('../../data/fm/train_ctr_pred.csv', header=None)

print(train_data)
print(train_ctrs)