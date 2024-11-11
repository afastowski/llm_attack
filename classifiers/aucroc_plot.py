import json
import matplotlib.pyplot as plt

with open('best_random_forest_results.json', 'r') as f:
    results = json.load(f)

models = {"gpt-4o": "GPT-4o", "gpt-4o-mini": "GPT-4o-mini", "mistral": "Mistral-7B", "llama": "LLaMA-2-13B", "phi": "Phi-3.5-mini"}
attack_types = [
    "No Attack vs. $\\mathcal{X}mera$", 
    "No Attack vs. $\\alpha$-$\\mathcal{X}mera$", 
    "No Attack vs. $\\beta$-$\\mathcal{X}mera$", 
    "No Attack vs. $\\gamma$-$\\mathcal{X}mera$"
]
titles = {
    "No Attack vs. $\\mathcal{X}mera$": r"  $\chi$",
    "No Attack vs. $\\alpha$-$\\mathcal{X}mera$": r"$\alpha\chi$",
    "No Attack vs. $\\beta$-$\\mathcal{X}mera$": r"$\beta\chi$",
    "No Attack vs. $\\gamma$-$\\mathcal{X}mera$": r"$\gamma\chi$"
}
colors = ['orange', 'red', 'magenta', 'purple']

fig = plt.figure(figsize=(12, 8))

positions = {
    "gpt-4o": [0.2, 0.55, 0.25, 0.35],       # x, y, width, height
    "gpt-4o-mini": [0.54, 0.55, 0.25, 0.35],
    "mistral": [0.05, 0.1, 0.25, 0.35],
    "llama": [0.375, 0.1, 0.25, 0.35],
    "phi": [0.7, 0.1, 0.25, 0.35]
}

for model, pos in positions.items():
    ax = fig.add_axes(pos)
    ax.set_title(models[model], fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate" if model == "mistral" else "", fontsize=10)
    
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
    ax.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)

plt.savefig("aucroc_all_models_2x3.pdf")
plt.show()
