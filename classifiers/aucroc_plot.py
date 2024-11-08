import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load the saved results
with open("best_extra_trees_results.json", "r") as f:
    results = json.load(f)

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 3, height_ratios=[3, 1.3], hspace=0.3, wspace=0.1)

ax_big = fig.add_subplot(gs[0, :])
ax_big.set_aspect('equal')
first_result = results[0]

fpr = np.array(first_result["fpr"])
tpr = np.array(first_result["tpr"])
roc_auc = first_result["roc_auc"]

ax_big.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
ax_big.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_big.set_xlim([0.0, 1.0])
ax_big.set_ylim([0.0, 1.05])
ax_big.set_xlabel('FPR', fontsize=16)
ax_big.set_ylabel('TPR', fontsize=16)
ax_big.set_title(first_result["title"], fontsize=16)
ax_big.legend(loc="lower right", fontsize=16)
ax_big.tick_params(axis='both', which='major', labelsize=12)

for idx, result in enumerate(results[1:4]):
    ax = fig.add_subplot(gs[1, idx])
    ax.set_aspect('equal')
    fpr = np.array(result["fpr"])
    tpr = np.array(result["tpr"])
    roc_auc = result["roc_auc"]

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(result["title"], fontsize=12)
    ax.legend(loc="lower right", fontsize=12)
    
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig("extra_trees_roc.pdf")
plt.show()
