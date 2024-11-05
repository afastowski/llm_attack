import matplotlib.pyplot as plt
import numpy as np

categories = ["GPT-4o", "GPT-4o-mini", "LLaMA-2-13B", "Mistral-7B", "Phi-3.5-mini"]
triviaqa_acc = [79.0, 67.6, 42.8, 50.2, 38.2]
hotpotqa_acc = [56.3, 42.8, 28.4, 33.6, 25.8]
nq_acc = [45.9, 45.3, 25.8, 26.6, 18.9]
parameters = [1800, 180, 13, 7, 3.8]

x = np.arange(len(categories))
width = 0.25
fig, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x - width, triviaqa_acc, width, label='TriviaQA', color='#e41a1c')
bars2 = ax1.bar(x, hotpotqa_acc, width, label='HotpotQA', color='#377eb8')
bars3 = ax1.bar(x + width, nq_acc, width, label='NQ', color='#4daf4a')


for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax2 = ax1.twinx()

ax2.plot(x, parameters, color='black', marker='.', linestyle='--', label='Parameters', linewidth=1)
ax2.set_yscale('log')
ax2.set_ylabel('Parameters (in billions)', color='black')

ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy across Datasets and Model Size')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)

ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.95)) 
ax2.legend(loc='upper right') 

plt.tight_layout()
plt.show()
plt.savefig("plots/model_performances.pdf")
