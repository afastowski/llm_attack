import os
import json
import numpy as np
from imblearn.over_sampling import ADASYN
import random

"""
Builds the datasets for the attack classifiers. 

v0 = unattacked, v1 = alpha, v2 = beta, v3 = gamma.

The folder /scores_1000 is not part of this repository, hence the data cannot be built 
by running this code locally. However, we provide the resulting datasets in classifiers/data/per_model.
"""

base_dir = '/scores_1000/'
output_dir = '/classifiers/data/per_model/'
os.makedirs(output_dir, exist_ok=True)

new_entries_per_model = {}
current_id = 0

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "uncertainties_baseline.json" or ("x" not in file and "baseline" not in file):
            file_path = os.path.join(root, file)
            model_name = file_path.split(os.sep)[3]

            if model_name not in new_entries_per_model:
                new_entries_per_model[model_name] = []

            with open(file_path, 'r') as f:
                version = "v0" if file == "uncertainties_baseline.json" else ""
                if "direct" in file_path:
                    version = "v1"
                elif "false" in file_path:
                    version = "v2"
                elif "random" in file_path:
                    version = "v3"
                data = json.load(f)
                for sample in data:
                    sample_class = 0 if file == "uncertainties_baseline.json" else 1
                    if sample_class == 0:
                        new_entries_per_model[model_name].append(
                            {
                                "id": current_id,
                                "class": sample_class,
                                "version": version,
                                "model": model_name,
                                "uc": {
                                    "ae": sample["uncertainty"]["ae"][0],
                                    "ppl": sample["uncertainty"]["ppl"][0],
                                    "ap": sample["uncertainty"]["ap"][0]
                                }
                            }
                        )
                    else:
                        successful_idx = -1
                        for i in range(len(sample["model_answer"])):
                            # we take a sample where the attack was successful
                            if sample["answer"].strip().lower() not in sample["model_answer"][i].strip().lower():
                                successful_idx = i
                                break
                        if successful_idx != -1:
                          new_entries_per_model[model_name].append(
                            {
                                "id": current_id,
                                "class": sample_class,
                                "version": version,
                                "model": model_name,
                                "uc": {
                                    "ae": sample["uncertainty"]["ae"][successful_idx],
                                    "ppl": sample["uncertainty"]["ppl"][successful_idx],
                                    "ap": sample["uncertainty"]["ap"][successful_idx]
                                }
                            }
                        )  
                    current_id += 1

# Generate specific attack datasets
for model_name, entries in new_entries_per_model.items():
    v0_entries = [entry for entry in entries if entry["version"] == "v0"]
    v1_entries = [entry for entry in entries if entry["version"] == "v1"]
    v2_entries = [entry for entry in entries if entry["version"] == "v2"]
    v3_entries = [entry for entry in entries if entry["version"] == "v3"]

    specific_attack_datasets = {
        "v0_v1": (v0_entries, v1_entries, "v1"),
        "v0_v2": (v0_entries, v2_entries, "v2"),
        "v0_v3": (v0_entries, v3_entries, "v3"),
    }

    for attack_type, (v0_data, attack_data, version_label) in specific_attack_datasets.items():
        
        # Split original data into test set
        test_v0 = random.sample(v0_data, min(200, len(v0_data)))
        test_attack = random.sample(attack_data, min(200, len(attack_data)))
        test_set = test_v0 + test_attack
        
        # Remaining original data is used for training
        train_v0 = [entry for entry in v0_data if entry not in test_v0]
        train_attack = [entry for entry in attack_data if entry not in test_attack]
        
        # Ensure 2500 samples per class using ADASYN if necessary
        X = np.array([[entry["uc"]["ae"], entry["uc"]["ppl"], entry["uc"]["ap"]] for entry in train_v0 + train_attack])
        y = np.array([0] * len(train_v0) + [1] * len(train_attack))
        
        adasyn = ADASYN(sampling_strategy={0: 2000, 1: 2000}, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        # Store augmented training data only
        train_set = []
        base_id = max(entry["id"] for entry in train_v0 + train_attack) + 1
        
        for i, (features, label) in enumerate(zip(X_resampled, y_resampled), start=base_id):
            entry = {
                "id": i,
                "class": int(label),
                "version": "v0" if label == 0 else version_label,
                "uc": {
                    "ae": features[0],
                    "ppl": features[1],
                    "ap": features[2]
                }
            }
            train_set.append(entry)
        
        output_path = os.path.join(output_dir, f"{model_name}/uncertainties_{attack_type}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({"train": train_set, "test": test_set}, f, indent=3)
        print(f"Dataset for {model_name} ({attack_type}) saved to {output_path}")