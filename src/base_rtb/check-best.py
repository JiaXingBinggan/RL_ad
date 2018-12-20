import sys
import random


setting_row = {}
setting_perf = {}

# setting is (proportion, algorithm)

fi = open('../../result/results.txt', 'r') # rtb.result.1458.txt
fo = open('../../result/results.txt'.replace('.txt', '.best.perf.txt'), 'w')
first = True
for line in fi:
    line = line.strip()
    s = line.split('\t')
    if first:
        first = False
        fo.write(line + '\n')
        continue
    algo = s[10]
    prop = s[0]
    perf = float(s[1]) # 选择点击2排序，利润1
    setting = (prop, algo)
    if setting in setting_perf and perf > setting_perf[setting] or setting not in setting_perf:
        setting_perf[setting] = perf
        setting_row[setting] = line
fi.close()
for setting in sorted(setting_perf):
    fo.write(setting_row[setting] + '\n')
fo.close()