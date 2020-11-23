import matplotlib.pyplot as plt
from pylab import *                                 #支持中文

'''
折线图（Point-1）- Recall rate
'''
mpl.rcParams['font.sans-serif'] = ['SimHei']

names = ['1', '5', '10', '15', '20']
x = range(len(names))
y = [0.0206, 0.0765, 0.1264, 0.1928, 0.2446]
y1 = [0.0658, 0.1837, 0.3358, 0.4274, 0.5013]
y_dataset2 = [0.0026, 0.0099, 0.0163, 0.0216, 0.0337]
y1_dataset2 = [0.0205, 0.0782, 0.1601, 0.2069, 0.2504]
# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11)  # 限定横轴的范围
# pl.ylim(-1, 110)  # 限定纵轴的范围
plt.plot(x, y_dataset2, marker='o', mec='r', mfc='w', label=u'Not using neural attention network and public preference')
plt.plot(x, y1_dataset2, marker='*', ms=10, label=u'Using neural attention network and public preference')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=5)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel("Recommended list length") #X轴标签
plt.ylabel("Recall rate") #Y轴标签
# plt.title("A simple plot") #标题

plt.show()
