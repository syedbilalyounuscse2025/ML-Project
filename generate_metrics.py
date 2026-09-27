import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/student_placement_synthetic.csv"
)

# Remove salary because it should not be used
if "salary_package_lpa" in df.columns:
    df = df.drop(columns=["salary_package_lpa"])

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("placement_status", axis=1)
y = df["placement_status"]

# ==========================================
# SAME TRAIN/TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# LOAD SAVED MODEL
# ==========================================

model = joblib.load(
    "model/best_model.pkl"
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(X_test)

probability = model.predict_proba(X_test)[:, 1]

# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    prediction
)

f1 = f1_score(
    y_test,
    prediction
)

roc_auc = roc_auc_score(
    y_test,
    probability
)

cm = confusion_matrix(
    y_test,
    prediction
)

print("=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# MODEL COMPARISON
# ==========================================

metrics = {

    "Logistic Regression": {
        "accuracy": 0.6268,
        "f1": 0.6397,
        "roc_auc": 0.6796
    },

    "Decision Tree": {
        "accuracy": 0.6021,
        "f1": 0.6161,
        "roc_auc": 0.6374
    },

    "Random Forest": {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc
    }

}

# ==========================================
# SAVE METRICS
# ==========================================

output = {

    "best_model": "Random Forest",

    "accuracy": accuracy,

    "f1_score": f1,

    "roc_auc": roc_auc,

    "confusion_matrix": cm.tolist(),

    "models": metrics

}

with open(
    "model/metrics.json",
    "w"
) as file:

    json.dump(
        output,
        file,
        indent=4
    )

print("\nMetrics saved to model/metrics.json")

print("=" * 60)