import string

import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-3）- Threshold of News Similarity - Recall@10
'''


def ToString(s):
    re = []
    t = s.split(" ")
    for st in t:
        re.append(st)
    return re


def ToFloat(s):
    re = []
    t = s.split(" ")
    for st in t:
        re.append(float(st))
    return re


mpl.rcParams['font.sans-serif'] = ['SimHei']

namess = '2 4 6 8 10'
names = ToString(namess)

x = range(len(names))
y1_x = '0.152 0.168 0.162 0.160 0.159'
y1 = ToFloat(y1_x)

plt.ylim(0.14,0.18)
plt.plot(x, y1, marker='o', mec='r', mfc='w', label=u'STSTM')


plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Number of Time Intervals") #X轴标签
plt.ylabel("Recall@10") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()




