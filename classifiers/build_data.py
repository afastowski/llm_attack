import os
import json
import numpy as np
from imblearn.over_sampling import ADASYN
import random

base_dir = '/scores_1000/'
output_dir = '/classifiers/data/per_model/'
os.makedirs(output_dir, exist_ok=True)

new_entries_per_model = {}
current_id = 0

# Load and process files
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
                    current_id += 1


def create_balanced_dataset_with_adasyn(entries, test_size=500):
    class_0_entries = [entry for entry in entries if entry["class"] == 0]
    class_1_entries = [entry for entry in entries if entry["class"] == 1]

    test_class_0 = random.sample(class_0_entries, test_size)
    test_class_1 = random.sample(class_1_entries, test_size)
    test_set = test_class_0 + test_class_1
    train_class_0 = [entry for entry in class_0_entries if entry not in test_class_0]
    train_class_1 = [entry for entry in class_1_entries if entry not in test_class_1]
    train_set = train_class_0 + train_class_1

    X_train = np.array([[entry['uc']['ae'], entry['uc']['ppl'], entry['uc']['ap']] for entry in train_set])
    y_train = np.array([entry['class'] for entry in train_set])

    # Apply ADASYN to balance the training set
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    augmented_train_set = []
    augmented_id = current_id
    for features, label in zip(X_resampled, y_resampled):
        augmented_train_set.append(
            {
                "id": augmented_id,
                "class": int(label),
                "version": "v0" if label == 0 else "v_attack",
                "uc": {
                    "ae": features[0],
                    "ppl": features[1],
                    "ap": features[2]
                }
            }
        )
        augmented_id += 1

    return {"train": augmented_train_set, "test": test_set}, [entry for entry in augmented_train_set if entry["class"] == 0], test_class_0


# Generate and save datasets per model
for model_name, entries in new_entries_per_model.items():
    v0_entries = [entry for entry in entries if entry["version"] == "v0"]
    v1_entries = [entry for entry in entries if entry["version"] == "v1"]
    v2_entries = [entry for entry in entries if entry["version"] == "v2"]
    v3_entries = [entry for entry in entries if entry["version"] == "v3"]

    # Generate and save the combined dataset (v0 (baseline) vs all attacks) for each model
    combined_dataset, augmented_v0_entries, v0_test_entries = create_balanced_dataset_with_adasyn(entries)
    combined_output_path = os.path.join(output_dir, f"{model_name}/{model_name}_uncertainties_v0_all.json")
    with open(combined_output_path, 'w') as f:
        json.dump(combined_dataset, f, indent=4)
    print(f"Combined train and test dataset for {model_name} saved to {combined_output_path}")

    # Generate and save specific attack datasets (baseline vs alpha, vs beta, vs gamma)
    specific_attack_datasets = {
        "v0_v1": (augmented_v0_entries, v1_entries, "v1"),
        "v0_v2": (augmented_v0_entries, v2_entries, "v2"),
        "v0_v3": (augmented_v0_entries, v3_entries, "v3"),
    }

    for attack_type, (v0_data, attack_data, version_label) in specific_attack_datasets.items():
        test_class_0 = v0_test_entries
        test_class_1 = random.sample(attack_data, min(500, len(attack_data)))
        test_set = test_class_0 + test_class_1

        X = np.array([[entry["uc"]["ae"], entry["uc"]["ppl"], entry["uc"]["ap"]] for entry in v0_data + attack_data])
        y = np.array([0] * len(v0_data) + [1] * len(attack_data))

        adasyn = ADASYN(sampling_strategy={1: len(v0_data)}, random_state=42)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)

        balanced_entries = []
        base_id = max(entry["id"] for entry in v0_data + attack_data) + 1
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

        output_path = os.path.join(output_dir, f"{model_name}/{model_name}_uncertainties_{attack_type}.json")
        with open(output_path, 'w') as f:
            json.dump({"train": balanced_entries, "test": test_set}, f, indent=4)
        print(f"Dataset for {model_name} ({attack_type}) saved to {output_path}")
