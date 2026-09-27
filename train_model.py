import pandas as pd
import joblib
import json
import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("model", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "dataset/student_placement_synthetic.csv"
)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\nTarget Distribution:")
print(df["placement_status"].value_counts())


# ============================================================
# FEATURES
# ============================================================

features = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skills",
    "dsa_score",
    "aptitude_score",
    "communication_skills",
    "internships",
    "certifications"
]

X = df[features]
y = df["placement_status"]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical = [
    "branch",
    "college_tier"
]


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical = [
    "cgpa",
    "backlogs",
    "coding_skills",
    "dsa_score",
    "aptitude_score",
    "communication_skills",
    "internships",
    "certifications"
]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical
        ),

        (
            "num",

            "passthrough",

            numerical
        )

    ]
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

}


# ============================================================
# STORAGE
# ============================================================

all_metrics = {}

best_model = None
best_f1 = -1
best_name = ""

best_predictions = None
best_cm = None


print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


# ============================================================
# TRAIN MODELS
# ============================================================

for name, classifier in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )

    ])


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )


    probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


    # --------------------------------------------------------
    # STORE METRICS
    # --------------------------------------------------------

    all_metrics[name] = {

        "accuracy":
            round(float(accuracy), 4),

        "precision":
            round(float(precision), 4),

        "recall":
            round(float(recall), 4),

        "f1":
            round(float(f1), 4),

        "roc_auc":
            round(float(roc_auc), 4),

        "confusion_matrix":
            cm.tolist()

    }


    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        best_model = pipeline

        best_name = name

        best_predictions = predictions

        best_cm = cm


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(

    best_model,

    "model/best_model.pkl"

)


# ============================================================
# GET BEST MODEL METRICS
# ============================================================

best_metrics = all_metrics[best_name]


# ============================================================
# CREATE FINAL METRICS JSON
# ============================================================

metrics_data = {

    "best_model":
        best_name,

    "accuracy":
        best_metrics["accuracy"],

    "precision":
        best_metrics["precision"],

    "recall":
        best_metrics["recall"],

    "f1_score":
        best_metrics["f1"],

    "roc_auc":
        best_metrics["roc_auc"],

    "best_f1":
        best_metrics["f1"],

    "confusion_matrix":
        best_metrics["confusion_matrix"],

    "models":
        all_metrics

}


with open(
    "model/metrics.json",
    "w"
) as file:

    json.dump(
        metrics_data,
        file,
        indent=4
    )


# ============================================================
# GRAPH 1 — TARGET DISTRIBUTION
# ============================================================

target_counts = df["placement_status"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Less Likely", "Likely"],
    [
        target_counts.get(0, 0),
        target_counts.get(1, 0)
    ]
)

plt.title(
    "Student Placement Distribution"
)

plt.xlabel(
    "Placement Status"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/target_distribution.png"
)

plt.close()


# ============================================================
# GRAPH 2 — MODEL COMPARISON
# ============================================================

model_names = list(all_metrics.keys())

accuracies = [
    all_metrics[name]["accuracy"] * 100
    for name in model_names
]

f1_scores = [
    all_metrics[name]["f1"] * 100
    for name in model_names
]

roc_scores = [
    all_metrics[name]["roc_auc"] * 100
    for name in model_names
]


x = range(len(model_names))

width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width for i in x],
    accuracies,
    width=width,
    label="Accuracy"
)

plt.bar(
    x,
    f1_scores,
    width=width,
    label="F1 Score"
)

plt.bar(
    [i + width for i in x],
    roc_scores,
    width=width,
    label="ROC-AUC"
)

plt.xticks(
    list(x),
    model_names
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.legend()

plt.ylim(
    0,
    100
)

plt.tight_layout()

plt.savefig(
    "static/graphs/model_comparison.png"
)

plt.close()


# ============================================================
# GRAPH 3 — FEATURE IMPORTANCE
# ============================================================

preprocessor_final = (
    best_model
    .named_steps["preprocessor"]
)

classifier_final = (
    best_model
    .named_steps["classifier"]
)

feature_names = (
    preprocessor_final
    .get_feature_names_out()
)


# Random Forest / Decision Tree
if hasattr(
    classifier_final,
    "feature_importances_"
):

    importances = (
        classifier_final
        .feature_importances_
    )


# Logistic Regression
elif hasattr(
    classifier_final,
    "coef_"
):

    importances = abs(
        classifier_final.coef_[0]
    )

else:

    importances = [
        0
        for _ in feature_names
    ]


feature_df = pd.DataFrame({

    "feature":
        feature_names,

    "importance":
        importances

})


feature_df = feature_df.sort_values(

    "importance",

    ascending=False

).head(10)


plt.figure(
    figsize=(10, 6)
)

plt.barh(

    feature_df["feature"][::-1],

    feature_df["importance"][::-1]

)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/feature_importance.png"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print(
    "BEST MODEL:",
    best_name
)

print(
    "Accuracy:",
    best_metrics["accuracy"]
)

print(
    "Precision:",
    best_metrics["precision"]
)

print(
    "Recall:",
    best_metrics["recall"]
)

print(
    "F1 Score:",
    best_metrics["f1"]
)

print(
    "ROC-AUC:",
    best_metrics["roc_auc"]
)

print("\nSaved files:")

print(
    "model/best_model.pkl"
)

print(
    "model/metrics.json"
)

print(
    "static/graphs/target_distribution.png"
)

print(
    "static/graphs/model_comparison.png"
)

print(
    "static/graphs/feature_importance.png"
)

print("=" * 70)