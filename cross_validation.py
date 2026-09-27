import pandas as pd
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

df = pd.read_csv(
    "dataset/student_placement_synthetic.csv"
)

if "salary_package_lpa" in df.columns:

    df = df.drop(
        columns=["salary_package_lpa"]
    )

X = df.drop(
    "placement_status",
    axis=1
)

y = df[
    "placement_status"
]

model = joblib.load(
    "model/best_model.pkl"
)

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)

scores = cross_val_score(

    model,

    X,

    y,

    cv=cv,

    scoring="f1",

    n_jobs=-1

)

print(
    "Cross Validation F1 Scores:"
)

print(scores)

print(
    "\nMean F1:",
    scores.mean()
)

print(
    "Standard Deviation:",
    scores.std()
)