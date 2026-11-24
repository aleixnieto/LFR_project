import json
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

path = Path(
    "evaluations/failure_analysis_ablation/baseline/predictions.json"
)

with open(path, "r") as f:
    data = json.load(f)

y_true = [item["ground_truth"] for item in data]
y_pred = [item["prediction"] for item in data]


classes_to_merge = {"Playground", "playfields"}
merged_label = "RecreationalArea"

y_true_merged = [
    merged_label if label in classes_to_merge else label
    for label in y_true
]

y_pred_merged = [
    merged_label if label in classes_to_merge else label
    for label in y_pred
]

print("Nombre d'images :", len(data))
print("Accuracy originale :", accuracy_score(y_true, y_pred))
print("Accuracy après fusion :", accuracy_score(y_true_merged, y_pred_merged))

print("\nRapport après fusion :")
print(classification_report(y_true_merged, y_pred_merged))