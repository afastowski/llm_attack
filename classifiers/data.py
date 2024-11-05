import os
import json

base_dir = 'scores_1000/'
output_file = 'classifiers/data/uncertainties_max_training_all_uc.json'

new_entries = []
current_base_length = 0
current_id = 0

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
                                "class": 0, # 0 because it's a non-attacked sample
                                "version": "v0",
                                "uc": {
                                    "ae": sample["uncertainty"]["ae"],
                                    "ppl": sample["uncertainty"]["ppl"],
                                    "ap": sample["uncertainty"]["ap"]
                                }
                            }
                        )
                    current_id += 1
        elif "x" not in file:  # going to all the other files, but ignoring the "false_x10" etc files here
            # Get the full file path
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
                            "class": 1, # 1 because it's an attacked sample
                            "version": version,
                            "uc": {
                                    "ae": sample["uncertainty"]["ae"],
                                    "ppl": sample["uncertainty"]["ppl"],
                                    "ap": sample["uncertainty"]["ap"]
                                }
                        }
                    )
                    current_id += 1

with open(output_file, 'w') as f:
    json.dump(new_entries, f, indent=4) 