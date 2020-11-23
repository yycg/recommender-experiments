import string

import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-2）- Number of Neurons per Layer - LOGLOSS
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
y1_x = '0.03897	0.03893	0.03892	0.03889'
y1 = ToFloat(y1_x)

y2_x = '0.03878	0.03877	0.03875	0.03873'
y2 = ToFloat(y2_x)

y3_x = '0.03849	0.03847	0.03845	0.03844'
y3 = ToFloat(y3_x)

y4_x = '0.03891	0.03887	0.03885	0.03882'
y4 = ToFloat(y4_x)

y5_x = '0.03821	0.03816	0.03814	0.03813'
y5 = ToFloat(y5_x)

y6_x = '0.03812	0.03807	0.03804	0.03802'
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
plt.ylabel("LOGLOSS") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()




