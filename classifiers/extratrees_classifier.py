import json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, classification_report

dataset_files = [
    'data/uncertainties_v0_all.json',
    'data/uncertainties_v0_v1.json',
    'data/uncertainties_v0_v2.json',
    'data/uncertainties_v0_v3.json'
]

titles = {
    'data/uncertainties_v0_all.json': r"No Attack vs. $\mathcal{X}mera$",
    'data/uncertainties_v0_v1.json': r"No Attack vs. $\alpha$-$\mathcal{X}mera$",
    'data/uncertainties_v0_v2.json': r"No Attack vs. $\beta$-$\mathcal{X}mera$",
    'data/uncertainties_v0_v3.json': r"No Attack vs. $\gamma$-$\mathcal{X}mera$"
}

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

all_results = []

for file_path in dataset_files:
    with open(file_path, 'r') as f:
        data = json.load(f)

    X = np.array([[entry['uc']['ae'], entry['uc']['ppl'], entry['uc']['ap']] for entry in data])
    y = np.array([entry['class'] for entry in data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    model = ExtraTreesClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        n_jobs=-1,
        cv=skf,
        verbose=1,
        return_train_score=False
    )

    print(f"\nTuning ExtraTreesClassifier for dataset: {titles[file_path]}")
    grid_search.fit(X_train, y_train_encoded)

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
        "title": titles[file_path],
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
    print("Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print("\nFinal Evaluation on True Test Set:")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test F1 Score: {test_f1}")
    print(f"ROC AUC: {roc_auc}")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_.astype(str)))
    print("-" * 50)

with open('best_extra_trees_results.json', 'w') as outfile:
    json.dump(all_results, outfile, indent=4)

print("\nAll tuning and test set results saved to 'best_extra_trees_results.json'")
