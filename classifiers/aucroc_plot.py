import json
import matplotlib.pyplot as plt


with open('best_random_forest_results.json', 'r') as f:
    results = json.load(f)

models = {"gpt-4o": "GPT-4o", "gpt-4o-mini": "GPT-4o-mini", "mistral": "Mistral-7B", "llama": "LLaMA-2-13B", "phi": "Phi-3.5-mini"}
attack_types = [
    "No Attack vs. $\\alpha$-$\\mathcal{X}mera$", 
    "No Attack vs. $\\beta$-$\\mathcal{X}mera$", 
    "No Attack vs. $\\gamma$-$\\mathcal{X}mera$"
]
titles = {
    "No Attack vs. $\\alpha$-$\\mathcal{X}mera$": r"$\alpha\chi$",
    "No Attack vs. $\\beta$-$\\mathcal{X}mera$": r"$\beta\chi$",
    "No Attack vs. $\\gamma$-$\\mathcal{X}mera$": r"$\gamma\chi$"
}
colors = ['darkorange', 'magenta', 'royalblue']

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, (model, model_name) in zip(axes, models.items()):
    ax.set_title(model_name, fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("False Positive Rate", fontsize=10)
    if model == "gpt-4o":
        ax.set_ylabel("True Positive Rate", fontsize=10)
    
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
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("plots/aucroc_all_models_1x5.pdf")
plt.show()