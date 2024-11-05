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

    # "PPL_Correct_Avg": [1.34, 1.07, 1.09, 1.32, 1.07, 1.08, 1.57, 1.16, 1.16, 1.31, 1.10, 1.10, 1.39, 1.13, 1.14],
    # "PPL_Incorrect_Avg": [1.6, 1.25, 1.35, 1.48, 1.2, 1.32, 1.77, 1.17, 1.31, 1.37, 1.12, 1.20, 1.48, 1.11, 1.30],

    # "TP_Correct_Avg": [0.81, 0.94, 0.93, 0.81, 0.95, 0.94, 0.73, 0.89, 0.9, 0.82, 0.92, 0.92, 0.78, 0.91, 0.9],
    # "TP_Incorrect_Avg": [0.67, 0.86, 0.82, 0.75, 0.88, 0.82, 0.68, 0.89, 0.81, 0.80, 0.91, 0.86, 0.75, 0.92, 0.82],


}

df = pd.DataFrame(data)

# Define the custom order for the "Model" column
model_order = ["GPT-4o", "GPT-4o-mini", "Mistral-7B", "LLaMA-2-13B", "Phi-3.5-mini"]

# Convert "Model" to a categorical type with the custom order
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

# Sort by both Version and Model
df_sorted = df.sort_values(by=["Version", "Model"])

# Define the versions
versions = [r'$\alpha$-$\mathcal{X}mera$', r'$\beta$-$\mathcal{X}mera$', r'$\gamma$-$\mathcal{X}mera$']

# Create subplots with 1 row and 3 columns (one for each version)
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

bar_width = 0.35
offset = bar_width / 2  # Offset by half the bar width for spacing

# Plot Entropy plots for each version
for col, version in enumerate(versions):
    # Filter the data for the current version
    df_version = df_sorted[df_sorted["Version"].apply(lambda x: x in version)]
    
    # Create the labels for each model
    models_sorted = df_version["Model"].astype(str)
    x = range(len(models_sorted))  # Define x positions for bars
    
    # Plot the correct and incorrect scores with offset
    bars1 = axes[col].bar([pos - offset for pos in x], df_version["H_Correct_Avg"], 
                          width=bar_width, label="Correct", alpha=0.7, align='center')
    bars2 = axes[col].bar([pos + offset for pos in x], df_version["H_Incorrect_Avg"], 
                          width=bar_width, label="Incorrect", alpha=0.7, align='center')
    
    # Set the title for each subplot with larger font size
    axes[col].set_title(f"{version}: Entropy", fontsize=16)
    
    # Set x-tick labels
    axes[col].set_xticks(x)
    axes[col].set_xticklabels(models_sorted, rotation=45, ha="right", fontsize=14)
    
    # Set the Y label for the first subplot only with larger font size
    if col == 0:
        axes[col].set_ylabel("Entropy", fontsize=16)

# Make y-tick labels larger
for ax in axes.flat:
    ax.tick_params(axis='y', labelsize=14)

handles, labels = axes[0].get_legend_handles_labels()
axes[1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize=14, frameon=True)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('drift/plots/entropy_only_correct_incorrect.pdf')