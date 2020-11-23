import matplotlib.pyplot as plt
import numpy as np


'''
条形图（Point-1）
'''
recall = [0.3858, 0.4191, 0.4503, 0.5972, 0.7088]
NDCG = [0.2305, 0.2514, 0.2880, 0.4274, 0.6308]
recall_dataset_2 = [0.4521,0.3062,0.3612,0.8254,0.9146]
NDCG_dataset_2 = [0.3003,0.1720,0.2477,0.7399,0.8508]
num_cols = ['embed(1,2,3,4,5)', 'embed(1,2,3,5)', 'embed(1,2,5)', 'embed(1,2,3)', 'embed(1,2)']
fig, ax = plt.subplots()
position = np.arange(1, 6)


ax.bar(position, recall_dataset_2, 0.5)
ax.set_xticks(range(1, 6))
ax.set_xticklabels(num_cols, rotation=10)
# ax.set_xlabel('Methods')
ax.set_ylabel('Rec@5')
plt.show()

# ax.bar(position, NDCG_dataset_2, 0.5)
# ax.set_xticks(range(1, 6))
# ax.set_xticklabels(num_cols, rotation=10)
# # ax.set_xlabel('Methods')
# ax.set_ylabel('NDCG@5')
# plt.show()