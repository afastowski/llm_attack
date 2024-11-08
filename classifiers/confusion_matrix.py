import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

with open("best_extra_trees_results.json", "r") as f:
    results = json.load(f)

num_classifiers = len(results)
rows = 2
cols = 2
fig, axes = plt.subplots(1, num_classifiers, figsize=(15, 5))
axes = axes.ravel()

for idx, result in enumerate(results):
    y_test = np.array(result["y_test"])
    y_pred = np.array(result["y_pred"])
    title = result["title"]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=axes[idx], values_format='d', colorbar=False)

    axes[idx].set_title(f"{title}\nF1: {f1:.2f}", fontsize=16)
    axes[idx].set_xlabel("Predicted label")
    axes[idx].set_ylabel("True label")

plt.tight_layout()
plt.suptitle("Confusion Matrices of All Classifiers", fontsize=16, y=1.05)
plt.savefig("extratrees_confusion_matrix.pdf")
