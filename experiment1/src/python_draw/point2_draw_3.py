import string

import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-2）- Number of Hidden Layers - AUC
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

namess = '2	4	6	8'
names = ToString(namess)

x = range(len(names))
y1_x = '0.7832	0.7839	0.7841	0.7838'
y1 = ToFloat(y1_x)

y2_x = '0.7872	0.7876	0.7879	0.7874'
y2 = ToFloat(y2_x)

y3_x = '0.7882	0.7889	0.7893	0.7891'
y3 = ToFloat(y3_x)

y4_x = '0.7848	0.7852	0.7858	0.7855'
y4 = ToFloat(y4_x)

y5_x = '0.7901	0.7909	0.7915	0.7911'
y5 = ToFloat(y5_x)

y6_x = '0.7904	0.7912	0.7921	0.7913'
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
plt.xlabel("Number of Hidden Layers") #X轴标签
plt.ylabel("AUC") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()



