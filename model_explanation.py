import joblib
import pandas as pd


MODEL_PATH = "model/best_model.pkl"

model = joblib.load(MODEL_PATH)


def get_feature_importance():

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Random Forest feature importance
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


if __name__ == "__main__":

    df = get_feature_importance()

    print("\nTop 15 Important Features:\n")

    print(
        df.head(15).to_string(index=False)
    )