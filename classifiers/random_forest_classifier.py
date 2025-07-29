import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report


base_dir = 'data/per_model'

titles = {
    "v0_all": r"No Attack vs. $\mathcal{X}mera$",
    "v0_v1": r"No Attack vs. $\alpha$-$\mathcal{X}mera$",
    "v0_v2": r"No Attack vs. $\beta$-$\mathcal{X}mera$",
    "v0_v3": r"No Attack vs. $\gamma$-$\mathcal{X}mera$"
}

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

all_results = []

for model_name in os.listdir(base_dir):
    model_path = os.path.join(base_dir, model_name)
    
    if not os.path.isdir(model_path):
        continue

    print(f"\nProcessing model: {model_name}")

    for dataset_key in titles.keys():
        dataset_file = f"{model_name}_uncertainties_{dataset_key}.json"
        dataset_path = os.path.join(model_path, dataset_file)

        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found. Skipping...")
            continue

        with open(dataset_path, 'r') as f:
            data = json.load(f)

        X = np.array([[entry['uc']['ae'], entry['uc']['ppl'], entry['uc']['ap']] for entry in data["train"]])
        y = np.array([entry['class'] for entry in data["train"]])

        X_test = np.array([[entry['uc']['ae'], entry['uc']['ppl'], entry['uc']['ap']] for entry in data["test"]])
        y_test = np.array([entry['class'] for entry in data["test"]])

        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y)
        y_test_encoded = label_encoder.transform(y_test)

        model = RandomForestClassifier(random_state=42)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1',
            n_jobs=-1,
            cv=skf,
            verbose=1,
            return_train_score=False
        )

        print(f"\nTuning RandomForestClassifier for {model_name} - {titles[dataset_key]}")
        grid_search.fit(X, y_train_encoded)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_f1_score = grid_search.best_score_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test_encoded, y_pred)
        test_f1 = f1_score(y_test_encoded, y_pred)
        roc_auc = roc_auc_score(y_test_encoded, y_prob)
        fpr, tpr, _ = roc_curve(y_test_encoded, y_prob)

        result = {
            "model_name": model_name,
            "title": titles[dataset_key],
            "best_params": best_params,
            "cross_val_f1": best_f1_score,
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": roc_auc,
            "y_test": y_test_encoded.tolist(),
            "y_pred": y_pred.tolist()
        }
        all_results.append(result)

        print(f"Best F1 Score (cross-validated): {best_f1_score}")
        # print("Best Hyperparameters:")
        # for param, value in best_params.items():
        #     print(f"  {param}: {value}")

        print("\nFinal Evaluation on True Test Set:")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test F1 Score: {test_f1}")
        print(f"ROC AUC: {roc_auc}")
        print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_.astype(str)))
        print("-" * 50)

with open('best_random_forest_per_model_results.json', 'w') as outfile:
    json.dump(all_results, outfile, indent=4)

print("\nAll tuning and test set results saved to 'best_random_forest_per_model_results.json'")

models = ["gpt-4o", "gpt-4o-mini", "mistral", "llama", "phi"]
datasets = ["v0_all", "v0_v1", "v0_v2", "v0_v3"]

f1_scores_dict = {f"{model}_{dataset}": None for model in models for dataset in datasets}
roc_aucs_dict = {f"{model}_{dataset}": None for model in models for dataset in datasets}

for result in all_results:
    model_name = result["model_name"]
    title = result["title"]
    
    key = f"{model_name}_{title}"
    
    f1_scores_dict[key] = result["test_f1"]
    roc_aucs_dict[key] = result["roc_auc"]