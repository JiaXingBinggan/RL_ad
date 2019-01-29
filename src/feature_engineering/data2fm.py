# _*_ coding: utf-8 _*_
import operator
import collections
from csv import DictReader
from datetime import datetime
import pandas as pd

train_path = '../../sample/20130606_train_sample.csv'
test_path = '../../sample/20130612_test_sample.csv'
train_fm = '../../data/fm/train.txt'
test_fm = '../../data/fm/test.txt'
vali_path = '../../data/fm/validation.csv'
feature_index = '../../data/fm/featindex.txt'

# field = ['weekday', 'hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
#          'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'usertag']
field = ['hour', 'useragent', 'IP', 'region', 'city', 'adexchange', 'domain', 'slotid', 'slotwidth',
         'slotheight', 'slotvisibility', 'slotformat', 'creative', 'keypage', 'usertag']

table = collections.defaultdict(lambda: 0)

# 为特征名建立编号, filed
def field_index(x):
    index = field.index(x)
    return index

def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

for f in field:
    getIndices(str(field_index(f))+':other')

feature_indices = set()
train_fm_data = []
for e, row in enumerate(DictReader(open(train_path)), start=1):
    temp_train_fm_data = []
    features = []
    for k, v in row.items():
        if k in field:
            if k == 'usertag':
                v = v[:5]
            if len(v) > 0:
                kv = k + '_' + v
                features.append('{0}:1'.format(getIndices(kv)))
                feature_indices.add(kv + '\t' + str(getIndices(kv)))
            else:
                kv = k + '_' + 'other'
                features.append('{0}:1'.format(getIndices(kv)))
    if e % 100000 == 0:
        print(datetime.now(), 'creating train.fm...', e)
    for val in features:
        temp_train_fm_data.append(val)
    temp_train_fm_data.append(row['click'])
    temp_train_fm_data.append(row['payprice'])
    temp_train_fm_data.append(row['hour'])
    train_fm_data.append(temp_train_fm_data)
train_fm_df = pd.DataFrame(data=train_fm_data)
train_fm_df.to_csv(train_fm, header=None, index=None)

test_fm_data = []
for t, row in enumerate(DictReader(open(test_path)), start=1):
    temp_test_fm_data = []
    features = []
    for k, v in row.items():
        if k in field:
            if k == 'usertag':
                v = v[:5]
            if len(v) > 0:
                kv = k + '_' + v
                if kv + '\t' + str(getIndices(kv)) in feature_indices:
                    features.append('{0}:1'.format(getIndices(kv)))
                else:
                    kv = k + '_' + 'other'
                    features.append('{0}:1'.format(getIndices(kv)))
                feature_indices.add(kv + '\t' + str(getIndices(kv)))
            else:
                kv = k + '_' + 'other'
                features.append('{0}:1'.format(getIndices(kv)))

    if t % 100000 == 0:
        print(datetime.now(), 'creating validation data and test.fm...', t)
    for val in features:
        temp_test_fm_data.append(val)
    temp_test_fm_data.append(row['click'])
    temp_test_fm_data.append(row['payprice'])
    temp_test_fm_data.append(row['hour'])
    test_fm_data.append(temp_test_fm_data)
test_fm_df = pd.DataFrame(data=test_fm_data)
test_fm_df.to_csv(test_fm, header=None, index=None)

featvalue = sorted(table.items(), key=operator.itemgetter(1))
fo = open(feature_index, 'w')
for t, fv in enumerate(featvalue, start=1):
    if t > len(field):
        k = fv[0].split('_')[0]
        idx = field_index(k)
        fo.write(str(idx) + ':' + fv[0] + '\t' + str(fv[1]) + '\n')
    else:
        fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
fo.close()

print(datetime.now(), '将train.txt与test.txt转化为可处理的矩阵')
# 将train.txt与test.txt转化为可处理的矩阵
train_data = pd.read_csv('../../data/fm/train.txt', header=None).values
total_feature = []
for i in range(len(train_data)):
    index_feature = []
    for k in range(len(train_data[i])):
        if k == 15 or k == 16 or k == 17:
            index_feature.append(train_data[i][k])
        else:
            feature_index = train_data[i][k].split(':')[0]
            index_feature.append(int(feature_index))
    total_feature.append(index_feature)
train_df = pd.DataFrame(data=total_feature)
train_df.to_csv('../../data/fm/train_fm.csv', header=None, index=None)

test_data = pd.read_csv('../../data/fm/test.txt', header=None).values
total_feature = []
for i in range(len(test_data)):
    index_feature = []
    for k in range(len(test_data[i])):
        if k == 15 or k == 16 or k == 17:
            index_feature.append(test_data[i][k])
        else:
            feature_index = test_data[i][k].split(':')[0]
            index_feature.append(int(feature_index))
    total_feature.append(index_feature)
train_df = pd.DataFrame(data=total_feature)
train_df.to_csv('../../data/fm/test_fm.csv', header=None, index=None)