import string

import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-3）- Threshold of News Similarity - NDCG@10
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

namess = '20 40 60 80 100 120'
names = ToString(namess)

x = range(len(names))
JIM_X = '0.64 0.63 0.59 0.62 0.61 0.65'
USTTM_X = '0.556 0.60 0.68 0.65 0.64 0.63'
STT_X = '0.45 0.52 0.54 0.53 0.55 0.54'
STSTM_X='0.68 0.67 0.70 0.65 0.69 0.68'
SPORE_X='0.334 0.337 0.348 0.396 0.376 0.482'
JIM = ToFloat(JIM_X)
USTTM = ToFloat(USTTM_X)
STT = ToFloat(STT_X)
STSTM = ToFloat(STSTM_X)
SPORE=ToFloat(SPORE_X)

plt.ylim(0.2,0.8)
plt.plot(x, JIM, marker='o', mec='r', mfc='w', label=u'JIM')
plt.plot(x, USTTM, marker='^', ms=10, label=u'USTTM')
plt.plot(x, STT, marker='x', mec='r', mfc='w', label=u'STT')
plt.plot(x, STSTM, marker='+', ms=10, label=u'STSTM')
plt.plot(x, SPORE, marker='*', ms=10, label=u'SPORE')

plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Number of Topic") #X轴标签
plt.ylabel("ndcg@10") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()




