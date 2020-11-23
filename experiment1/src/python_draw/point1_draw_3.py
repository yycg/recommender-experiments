import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']

'''
折线图（Point-1）- Hit
'''

names = ['1', '5', '10', '15', '20']
x = range(len(names))
y = [0.0230, 0.0171, 0.0135, 0.0118, 0.0117]
y1 = [0.0658, 0.0495, 0.0355, 0.0305, 0.0263]
y_dataset2 = [0.0019, 0.0024, 0.0021, 0.0017, 0.0021]
y1_dataset2 = [0.0205, 0.0147, 0.0163, 0.0140, 0.0128]
# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'Not using neural attention network and public preference')
plt.plot(x, y1, marker='*', ms=10, label=u'Using neural attention network and public preference')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Recommended list length") #X轴标签
plt.ylabel("Hit") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()
