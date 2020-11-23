import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

# precision
algorithms = ["a", "b"]
ndcg_at_10s = [0, 0.5]
ndcg_at_20s = [0, 0.5]
ndcg_at_50s = [0, 0.5]

fig, ax = plt.subplots(1, 3)
positions = np.arange(3)
width = 0.05
for i, algorithm in enumerate(algorithms):
    ndcgs = [ndcg_at_10s[i], ndcg_at_20s[i], ndcg_at_50s[i]]
    ax[0].bar(positions+i*width, ndcgs, width, label=algorithm)
ax[0].set_xticks(positions+len(algorithms)*width/2)
ax[0].set_xticklabels(["10", "20", "50"])
ax[0].set_xlabel("TopK")
ax[0].set_ylabel("nDCG")
ax[0].legend(loc="best")

# recall
algorithms = ["a", "b"]
ndcg_at_10s = [0, 0.5]
ndcg_at_20s = [0, 0.5]
ndcg_at_50s = [0, 0.5]

positions = np.arange(3)
width = 0.05
for i, algorithm in enumerate(algorithms):
    ndcgs = [ndcg_at_10s[i], ndcg_at_20s[i], ndcg_at_50s[i]]
    ax[1].bar(positions+i*width, ndcgs, width, label=algorithm)
ax[1].set_xticks(positions+len(algorithms)*width/2)
ax[1].set_xticklabels(["10", "20", "50"])
ax[1].set_xlabel("TopK")
ax[1].set_ylabel("nDCG")
ax[1].legend(loc="best")

# ndcg
algorithms = ["a", "b"]
ndcg_at_10s = [0, 0.5]
ndcg_at_20s = [0, 0.5]
ndcg_at_50s = [0, 0.5]

positions = np.arange(3)
width = 0.05
for i, algorithm in enumerate(algorithms):
    ndcgs = [ndcg_at_10s[i], ndcg_at_20s[i], ndcg_at_50s[i]]
    ax[2].bar(positions+i*width, ndcgs, width, label=algorithm)
ax[2].set_xticks(positions+len(algorithms)*width/2)
ax[2].set_xticklabels(["10", "20", "50"])
ax[2].set_xlabel("TopK")
ax[2].set_ylabel("nDCG")
ax[2].legend(loc="best")

fig.savefig('./images/evaluations.eps', dpi=600, format="eps")
fig.savefig("./images/evaluations.png", dpi=600, format="png")
