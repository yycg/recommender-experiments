import string

import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-2）- Number of Neurons per Layer - AUC
'''


def ToString(s):
    re = []
    t = s.split("\t")
    for st in t:
        re.append(st)
    return re


def ToFloat(s):
    re = []
    t = s.split("\t")
    for st in t:
        re.append(float(st))
    return re


mpl.rcParams['font.sans-serif'] = ['SimHei']

namess = '100	200	400	800'
names = ToString(namess)

x = range(len(names))
y1_x = '0.7836	0.7838	0.7839	0.7841'
y1 = ToFloat(y1_x)

y2_x = '0.7872	0.7873	0.7877	0.7879'
y2 = ToFloat(y2_x)

y3_x = '0.7882	0.7886	0.7891	0.7893'
y3 = ToFloat(y3_x)

y4_x = '0.7848	0.7853	0.7856	0.7858'
y4 = ToFloat(y4_x)

y5_x = '0.7901	0.7905	0.7911	0.7915'
y5 = ToFloat(y5_x)

y6_x = '0.7908	0.7912	0.7918	0.7921'
y6 = ToFloat(y6_x)

plt.plot(x, y1, marker='o', mec='r', mfc='w', label=u'LR')
plt.plot(x, y2, marker='^', ms=10, label=u'AFM')
plt.plot(x, y3, marker='+', ms=10, label=u'FFM')
plt.plot(x, y4, marker='.', ms=10, label=u'FM')
plt.plot(x, y5, marker='4', ms=10, label=u'DFM')
plt.plot(x, y6, marker='*', ms=10, label=u'MyModel')


plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Number of Neurons per Layer") #X轴标签
plt.ylabel("AUC") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()




