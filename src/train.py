# src/train.py
import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.data_cleaning import build_preprocessor

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Load Iris into a DataFrame and return (df, target_column_name)."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.rename(columns={"target": "species"}, inplace=True)
    return df, "species"


def make_model():
    """Return a small, reliable baseline model."""
    return LogisticRegression(max_iter=1000)


def train():
    df, target = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]),
        df[target],
        test_size=0.2,
        random_state=42,
        stratify=df[target],
    )

    preprocessor = build_preprocessor(df, target)
    model = make_model()
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    train()
