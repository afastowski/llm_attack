import matplotlib.pyplot as plt
import pandas as pd


metric = "TP"

data = {
    "Model": [
        "GPT-4o", "GPT-4o", "GPT-4o",
        "GPT-4o-mini", "GPT-4o-mini", "GPT-4o-mini",
        "Mistral-7B", "Mistral-7B", "Mistral-7B",
        "LLaMA-2-13B", "LLaMA-2-13B", "LLaMA-2-13B",
        "Phi-3.5-mini", "Phi-3.5-mini", "Phi-3.5-mini",
    ],
    "Version": ["alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha", "beta", "gamma"],
    
    "H_Correct_Avg": [0.66, 0.17, 0.20, 0.73, 0.17, 0.19, 0.68, 0.30, 0.29, 0.49, 0.21, 0.24, 0.59, 0.25, 0.27],
    "H_Incorrect_Avg": [1.14, 0.45, 0.59, 0.92, 0.44, 0.56, 0.78, 0.29, 0.48, 0.51, 0.23, 0.36, 0.64, 0.21, 0.48],

    "PPL_Correct_Avg": [1.34, 1.07, 1.09, 1.32, 1.07, 1.08, 1.57, 1.16, 1.16, 1.31, 1.10, 1.10, 1.39, 1.13, 1.14],
    "PPL_Incorrect_Avg": [1.6, 1.25, 1.35, 1.48, 1.2, 1.32, 1.77, 1.17, 1.31, 1.37, 1.12, 1.20, 1.48, 1.11, 1.30],

    "TP_Correct_Avg": [0.81, 0.94, 0.93, 0.81, 0.95, 0.94, 0.73, 0.89, 0.9, 0.82, 0.92, 0.92, 0.78, 0.91, 0.9],
    "TP_Incorrect_Avg": [0.67, 0.86, 0.82, 0.75, 0.88, 0.82, 0.68, 0.89, 0.81, 0.80, 0.91, 0.86, 0.75, 0.92, 0.82],


}
df = pd.DataFrame(data)

# ---- Preparation ----
model_order = ["GPT-4o", "GPT-4o-mini", "Mistral-7B", "LLaMA-2-13B", "Phi-3.5-mini"]
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
df_sorted = df.sort_values(by=["Version", "Model"])

versions = ["alpha", "beta", "gamma"]
version_titles = [r'$\alpha$-$\mathcal{X}mera$', r'$\beta$-$\mathcal{X}mera$', r'$\gamma$-$\mathcal{X}mera$']
metrics = [
    ("H_Correct_Avg", "H_Incorrect_Avg", "Entropy"),
    ("PPL_Correct_Avg", "PPL_Incorrect_Avg", "PPL"),
    ("TP_Correct_Avg", "TP_Incorrect_Avg", "TP"),
]

# ---- Plotting ----
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharey='row')
bar_width = 0.35
offset = bar_width / 2

for row, (correct_col, incorrect_col, title) in enumerate(metrics):
    for col, version in enumerate(versions):
        ax = axes[row][col]
        df_version = df_sorted[df_sorted["Version"] == version]
        models_sorted = df_version["Model"].astype(str)
        x = range(len(models_sorted))

        bars1 = ax.bar([pos - offset for pos in x], df_version[correct_col], 
                       width=bar_width, label="Correct", alpha=0.7)
        bars2 = ax.bar([pos + offset for pos in x], df_version[incorrect_col], 
                       width=bar_width, label="Incorrect", alpha=0.7)

        if row == 0:
            ax.set_title(version_titles[col], fontsize=16)
        if row == 2:
            ax.set_xticks(x)
            ax.set_xticklabels(models_sorted, rotation=45, ha="right", fontsize=12)
        else:
            ax.set_xticks([])  # hide x-ticks for upper rows

        if col == 0:
            ax.set_ylabel(title, fontsize=16)

        ax.tick_params(axis='y', labelsize=12)

# Add legend only once
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=14, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for legend
plt.savefig("plots/combined_metrics.pdf")