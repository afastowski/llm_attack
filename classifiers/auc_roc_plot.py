import json
import matplotlib.pyplot as plt

with open('classifiers/best_random_forest_results.json', 'r') as f:
    results = json.load(f)

models = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "mistral": "Mistral-7B",
    "llama": "LLaMA-2-13B",
    "phi": "Phi-3.5-mini"
}
model_order = ["gpt-4o", "gpt-4o-mini", "mistral", "llama", "phi"]

attack_types = [
    "No Attack vs. $\\mathcal{X}mera$", 
    "No Attack vs. $\\alpha$-$\\mathcal{X}mera$", 
    "No Attack vs. $\\beta$-$\\mathcal{X}mera$", 
    "No Attack vs. $\\gamma$-$\\mathcal{X}mera$"
]
titles = {
    "No Attack vs. $\\mathcal{X}mera$": r"$\chi$",
    "No Attack vs. $\\alpha$-$\\mathcal{X}mera$": r"$\alpha\chi$",
    "No Attack vs. $\\beta$-$\\mathcal{X}mera$": r"$\beta\chi$",
    "No Attack vs. $\\gamma$-$\\mathcal{X}mera$": r"$\gamma\chi$"
}
colors = ['orange', 'red', 'magenta', 'purple']

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

for ax, model in zip(axes, model_order):
    ax.set_title(models[model], fontsize=14)
    ax.set_xlabel("FPR", fontsize=12)
    if model == "gpt-4o":
        ax.set_ylabel("TPR", fontsize=12)

    ax.tick_params(axis='both', labelsize=12)
    
    for attack, color in zip(attack_types, colors):
        result = next(
            (res for res in results if res['model_name'] == model and attack in res['title']), None
        )
        if result:
            fpr = result['fpr']
            tpr = result['tpr']
            roc_auc = result['roc_auc']
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{titles[attack]} AUC = {roc_auc:.2f}")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
    ax.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("classifiers/new/aucroc_all_models_1x5.pdf")
plt.show()
