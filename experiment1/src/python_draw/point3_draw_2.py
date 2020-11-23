import string

import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-3）- The Number of Group - NDCG@10
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


namess = '20 40 60 80 100 120'
names = ToString(namess)

x = range(len(names))
JIM_X = '0.023 0.054 0.078 0.084 0.081 0.075'
USTTM_X = '0.04 0.060 0.089 0.118 0.088 0.081'
STT_X = '0.04 0.045 0.055 0.065 0.07 0.072'
STSTM_X='0.041 0.071 0.122 0.161 0.141 0.142'
JIM = ToFloat(JIM_X)
USTTM = ToFloat(USTTM_X)
STT = ToFloat(STT_X)
STSTM = ToFloat(STSTM_X)


plt.ylim(0.02,0.20)
plt.plot(x, JIM, marker='o', mec='r', mfc='w', label=u'JIM')
plt.plot(x, USTTM, marker='^', ms=10, label=u'USTTM')
plt.plot(x, STT, marker='o', mec='r', mfc='w', label=u'STT')
plt.plot(x, STSTM, marker='^', ms=10, label=u'STSTM')



plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("The Number of Region") #X轴标签
plt.ylabel("Recall@10") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()




