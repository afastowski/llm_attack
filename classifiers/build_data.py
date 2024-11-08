import os
import json
import numpy as np
from imblearn.over_sampling import ADASYN

base_dir = 'drift/scores_1000/'
output_file = 'drift/classifiers/data/uncertainties_v0_all.json'

new_entries = []
current_base_length = 0
current_id = 0

# Build initial dataset
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "uncertainties_baseline.json":
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                data = json.load(f) 
                current_base_length = len(data)
                for sample in data:
                    new_entries.append(
                        {
                            "id": current_id,
                            "class": 0,  # 0 because it's a non-attacked sample
                            "version": "v0",
                            "uc": {
                                "ae": sample["uncertainty"]["ae"][0],
                                "ppl": sample["uncertainty"]["ppl"][0],
                                "ap": sample["uncertainty"]["ap"][0]
                            }
                        }
                    )
                    current_id += 1
        elif "x" not in file:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                version = ""
                if "direct" in file_path:
                    version = "v1"
                elif "false" in file_path:
                    version = "v2"
                elif "random" in file_path:
                    version = "v3"
                data = json.load(f)
                for sample in data:
                    new_entries.append(
                        {
                            "id": current_id,
                            "class": 1,  # 1 because it's an attacked sample
                            "version": version,
                            "uc": {
                                "ae": sample["uncertainty"]["ae"][0],
                                "ppl": sample["uncertainty"]["ppl"][0],
                                "ap": sample["uncertainty"]["ap"][0]
                            }
                        }
                    )
                    current_id += 1

# Augment data with ADASYN
X = np.array([[entry["uc"]["ae"], entry["uc"]["ppl"], entry["uc"]["ap"]] for entry in new_entries])
y = np.array([entry["class"] for entry in new_entries])

adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

resampled_entries = []
for i, (features, label) in enumerate(zip(X_resampled, y_resampled)):
    entry = {
        "id": i,
        "class": int(label),
        "version": "v0" if label == 0 else new_entries[i]["version"],
        "uc": {
            "ae": features[0],
            "ppl": features[1],
            "ap": features[2]
        }
    }
    resampled_entries.append(entry)

with open(output_file, 'w') as f:
    json.dump(resampled_entries, f, indent=4)

print(f"Resampled dataset saved to {output_file}")

# ADASYN for the 3 specific classifiers

output_file_v0_v1 = 'drift/classifiers/data/uncertainties_v0_v1.json'
output_file_v0_v2 = 'drift/classifiers/data/uncertainties_v0_v2.json'
output_file_v0_v3 = 'drift/classifiers/data/uncertainties_v0_v3.json'

v0_entries = [entry for entry in resampled_entries if entry["version"] == "v0"]
v1_entries = [entry for entry in new_entries if entry["version"] == "v1"]
v2_entries = [entry for entry in new_entries if entry["version"] == "v2"]
v3_entries = [entry for entry in new_entries if entry["version"] == "v3"]


def create_balanced_dataset(v0_data, minority_data, version_label):
    X = np.array([[entry["uc"]["ae"], entry["uc"]["ppl"], entry["uc"]["ap"]] for entry in v0_data + minority_data])
    y = np.array([0] * len(v0_data) + [1] * len(minority_data))

    # Apply ADASYN to balance the minority class with v0
    adasyn = ADASYN(sampling_strategy={1: len(v0_data)}, random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    balanced_entries = []
    base_id = max(entry["id"] for entry in resampled_entries) + 1
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
        balanced_entries.append(entry)
    
    return balanced_entries

# Dataset 1: v0 vs v1 (baseline vs alpha)
balanced_v0_v1 = create_balanced_dataset(v0_entries, v1_entries, "v1")
with open(output_file_v0_v1, 'w') as f:
    json.dump(balanced_v0_v1, f, indent=4)
print(f"Dataset v0 vs v1 saved to {output_file_v0_v1}")

# Dataset 2: v0 vs v2 (baseline vs beta)
balanced_v0_v2 = create_balanced_dataset(v0_entries, v2_entries, "v2")
with open(output_file_v0_v2, 'w') as f:
    json.dump(balanced_v0_v2, f, indent=4)
print(f"Dataset v0 vs v2 saved to {output_file_v0_v2}")

# Dataset 3: v0 vs v3 (baseline vs gamma)
balanced_v0_v3 = create_balanced_dataset(v0_entries, v3_entries, "v3")
with open(output_file_v0_v3, 'w') as f:
    json.dump(balanced_v0_v3, f, indent=4)
print(f"Dataset v0 vs v3 saved to {output_file_v0_v3}")

