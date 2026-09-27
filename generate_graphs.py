import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# ==========================================
# CREATE GRAPH FOLDER
# ==========================================

os.makedirs(
    "static/graphs",
    exist_ok=True
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/student_placement_synthetic.csv"
)

# ==========================================
# GRAPH 1
# TARGET DISTRIBUTION
# ==========================================

counts = df[
    "placement_status"
].value_counts()

plt.figure(
    figsize=(8, 5)
)

counts.sort_index().plot(
    kind="bar"
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

plt.xticks(
    [0, 1],
    ["Not Placed", "Placed"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "static/graphs/target_distribution.png"
)

plt.close()

# ==========================================
# GRAPH 2
# MODEL COMPARISON
# ==========================================

with open(
    "model/metrics.json",
    "r"
) as file:

    metrics = json.load(file)

models = list(
    metrics["models"].keys()
)

accuracy = [
    metrics["models"][m]["accuracy"]
    for m in models
]

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    models,
    accuracy
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "static/graphs/model_comparison.png"
)

plt.close()

# ==========================================
# GRAPH 3
# FEATURE IMPORTANCE
# ==========================================

import joblib

model = joblib.load(
    "model/best_model.pkl"
)

preprocessor = model.named_steps[
    "preprocessor"
]

classifier = model.named_steps[
    "classifier"
]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importances = (
    classifier
    .feature_importances_
)

importance_df = pd.DataFrame({

    "feature": feature_names,

    "importance": importances

})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .head(10)
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    importance_df["feature"][::-1],
    importance_df["importance"][::-1]
)

plt.title(
    "Top 10 Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    "static/graphs/feature_importance.png"
)

plt.close()

print(
    "Graphs generated successfully!"
)