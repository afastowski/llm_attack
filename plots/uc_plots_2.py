# Here's a Python script that reads a JSON file with the described structure,
# calculates the average "ae" uncertainty score for each sample, and plots these scores.

import json
import matplotlib.pyplot as plt
from collections import Counter

# Function to load JSON data from a file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to calculate average "ae" uncertainty for each sample
def calculate_average_ae(data):
    average_ae_scores = []
    colors=[]
    for sample in data:
        ae_scores = sample['uncertainty']['ae']
        average_ae = sum(ae_scores) / len(ae_scores)
        average_ae_scores.append(average_ae)
        counter = Counter(sample["model_answer"])
        model_answer, frequency = counter.most_common(1)[0]
        correct_answer = sample["answer"]
        if correct_answer.lower() in model_answer.lower():
            color = "green"
            colors.append(color)
        else:
            color = "red"
            colors.append(color)
    return average_ae_scores, colors

# Function to plot the average "ae" uncertainty scores
def plot_ae_scores(scores, colors):
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(scores)), scores, color=colors, marker='o')
    plt.title('Average Uncertainty (ae) Scores for Each Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Average ae Score')
    plt.grid(True)
    plt.show()
    plt.savefig("uc_plot_2.png")

# Example usage:
# Assuming your JSON file is named "data.json" and located in the same directory
file_path = '/home/alinafastowski/projects/qa_investigations/drift/scores_1000/triviaqa/gpt-4o/v1/uncertainties_direct_prompt.json'  # Replace with your actual file path

data = load_json_file(file_path)

average_ae_scores, color = calculate_average_ae(data)
plot_ae_scores(average_ae_scores, color)

# This code reads the file, computes the average uncertainty, and plots it. You can adjust the file path as needed.
