import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "dataset/student_placement_synthetic.csv"
)

if "salary_package_lpa" in df.columns:

    df = df.drop(
        columns=["salary_package_lpa"]
    )

# ==========================================
# FEATURES
# ==========================================

X = df.drop(
    "placement_status",
    axis=1
)

y = df[
    "placement_status"
]

categorical = [
    "branch",
    "college_tier"
]

numerical = [
    col
    for col in X.columns
    if col not in categorical
]

# ==========================================
# PREPROCESSING
# ==========================================

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

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

# ==========================================
# OPTIMIZED RANDOM FOREST
# ==========================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)

pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        model
    )

])

# ==========================================
# TRAIN
# ==========================================

print("Training optimized model...")

pipeline.fit(
    X_train,
    y_train
)

# ==========================================
# TEST
# ==========================================

prediction = pipeline.predict(
    X_test
)

probability = pipeline.predict_proba(
    X_test
)[:, 1]

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

print("\n==============================")

print(
    f"Optimized Accuracy : {accuracy:.4f}"
)

print(
    f"Optimized F1       : {f1:.4f}"
)

print(
    f"Optimized ROC-AUC  : {roc_auc:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        prediction
    )
)

print("==============================")

# ==========================================
# SAVE
# ==========================================

joblib.dump(
    pipeline,
    "model/optimized_model.pkl"
)

print(
    "\nOptimized model saved!"
)