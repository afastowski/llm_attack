import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# --- Original data ---
data = {
    "Model": [
        "GPT-4o", "GPT-4o", "GPT-4o",
        "GPT-4o-mini", "GPT-4o-mini", "GPT-4o-mini",
        "Mistral-7B", "Mistral-7B", "Mistral-7B",
        "LLaMA-2-13B", "LLaMA-2-13B", "LLaMA-2-13B",
        "Phi-3.5-mini", "Phi-3.5-mini", "Phi-3.5-mini",
    ],
    "$\\mathcal{X}mera$ attack": ["$\\alpha$-$\\mathcal{X}mera$", "$\\beta$-$\\mathcal{X}mera$", "$\\gamma$-$\\mathcal{X}mera$"] * 5,

    "H_Correct_Avg": [0.66, 0.17, 0.20, 0.73, 0.17, 0.19, 0.68, 0.30, 0.29, 0.49, 0.21, 0.24, 0.59, 0.25, 0.27],
    "H_Incorrect_Avg": [1.14, 0.45, 0.59, 0.92, 0.44, 0.56, 0.78, 0.29, 0.48, 0.51, 0.23, 0.36, 0.64, 0.21, 0.48],

    "PPL_Correct_Avg": [1.34, 1.07, 1.09, 1.32, 1.07, 1.08, 1.57, 1.16, 1.16, 1.31, 1.10, 1.10, 1.39, 1.13, 1.14],
    "PPL_Incorrect_Avg": [1.6, 1.25, 1.35, 1.48, 1.2, 1.32, 1.77, 1.17, 1.31, 1.37, 1.12, 1.20, 1.48, 1.11, 1.30],

    "TP_Correct_Avg": [0.81, 0.94, 0.93, 0.81, 0.95, 0.94, 0.73, 0.89, 0.9, 0.82, 0.92, 0.92, 0.78, 0.91, 0.9],
    "TP_Incorrect_Avg": [0.67, 0.86, 0.82, 0.75, 0.88, 0.82, 0.68, 0.89, 0.81, 0.80, 0.91, 0.86, 0.75, 0.92, 0.82],
}

df = pd.DataFrame(data)

# --- Compute absolute differences ---
df["Entropy"] = (df["H_Correct_Avg"] - df["H_Incorrect_Avg"]).abs()
df["Perplexity"] = (df["PPL_Correct_Avg"] - df["PPL_Incorrect_Avg"]).abs()
df["Token Probability"] = (df["TP_Correct_Avg"] - df["TP_Incorrect_Avg"]).abs()

# --- Reshape for plotting ---
melted_diff = df.melt(
    id_vars=["Model", "$\\mathcal{X}mera$ attack"],
    value_vars=["Entropy", "Perplexity", "Token Probability"],
    var_name="Metric",
    value_name="Difference"
)

# --- Prepare grouped data ---
grouped = melted_diff.groupby(["Metric", "Model", "$\\mathcal{X}mera$ attack"]).sum().reset_index()

# --- Constants ---
metrics = ["Entropy", "Perplexity", "Token Probability"]
versions = ["$\\alpha$-$\\mathcal{X}mera$", "$\\beta$-$\\mathcal{X}mera$", "$\\gamma$-$\\mathcal{X}mera$"]
model_order = ["GPT-4o", "GPT-4o-mini", "Mistral-7B", "LLaMA-2-13B", "Phi-3.5-mini"]

color_map = {
    "$\\alpha$-$\\mathcal{X}mera$": "red",  # blue
    "$\\beta$-$\\mathcal{X}mera$": "magenta",  # orange
    "$\\gamma$-$\\mathcal{X}mera$": "purple"  # green
}

hatches = {
    "$\\alpha$-$\\mathcal{X}mera$": "/",
    "$\\beta$-$\\mathcal{X}mera$": "\\",
    "$\\gamma$-$\\mathcal{X}mera$": "x"
}

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), sharey=False)
bar_width = 0.2
x_indices = list(range(len(model_order)))

for ax, metric in zip(axes, metrics):
    for i, version in enumerate(versions):
        subset = grouped[(grouped["Metric"] == metric) & (grouped["$\\mathcal{X}mera$ attack"] == version)]
        subset = subset.set_index("Model").reindex(model_order)
        x = [idx + i * bar_width for idx in x_indices]

        ax.bar(x, subset["Difference"], width=bar_width,
               edgecolor=color_map[version],
               color='white',
               hatch=hatches[version],
               linewidth=1.2,
               label=version)

    ax.set_title(metric, fontsize=16)
    ax.set_xticks([r + bar_width for r in x_indices])
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.tick_params(axis='x', labelsize=14)
    if metric == "Entropy":
        ax.set_ylabel("Absolute Difference", fontsize=14)

# --- Legend ---
handles = [mpatches.Patch(facecolor='white', edgecolor=color_map[v], hatch=hatches[v],
                          label=v, linewidth=1.2) for v in versions]
axes[-1].legend(handles=handles, title="$\\mathcal{X}mera$ attack", loc="upper right", fontsize=14, title_fontsize=14)

plt.tight_layout()
plt.savefig("plots/difference_metrics.pdf")