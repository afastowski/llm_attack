import json
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

file_path = 'classifiers/data/uncertainties_max_training_all_uc.json'
with open(file_path, 'r') as f:
    data = json.load(f)

def prepare_data(data, version_a=None, version_b=None, use_class=False):
    X = []
    y = []
    for entry in data:
        uc_scores = entry['uc']
        if use_class:
            for i in range(len(uc_scores["ae"])):
                features = [uc_scores["ae"][i], uc_scores["ppl"][i], uc_scores["ap"][i]]
                X.append(features)
                y.append(entry['class'])
        else:
            version = entry["version"]
            if version == version_a or version == version_b:
                for i in range(len(uc_scores["ae"])):
                    features = [uc_scores["ae"][i], uc_scores["ppl"][i], uc_scores["ap"][i]]
                    X.append(features)
                    y.append(version)
    X = np.array(X)
    y = np.array(y)
    return X, y

classifiers_info = [
    {"use_class": True, "version_a": None, "version_b": None, "title": r"No Attack vs. $\mathcal{X}mera$"},
    {"use_class": False, "version_a": "v0", "version_b": "v1", "title": r"No Attack vs. $\alpha$-$\mathcal{X}mera$"},
    {"use_class": False, "version_a": "v0", "version_b": "v2", "title": r"No Attack vs. $\beta$-$\mathcal{X}mera$"},
    {"use_class": False, "version_a": "v0", "version_b": "v3", "title": r"No Attack vs. $\gamma$-$\mathcal{X}mera$"},
]

results = []

for clf_info in classifiers_info:
    X, y = prepare_data(data, version_a=clf_info["version_a"], version_b=clf_info["version_b"], use_class=clf_info["use_class"])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True, random_state=42)

    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier: {clf_info['title']}")
    print(f"Accuracy: {accuracy:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    results.append({
        "title": clf_info["title"],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc
    })

with open("classifiers/xgboost_results.json", "w") as f:
    json.dump(results, f)