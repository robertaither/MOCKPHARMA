import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pharma_trial_ml.config import (
    DATA_PATH,
    EXPERIMENT_NAME,
    METADATA_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_PATH,
)
from pharma_trial_ml.data import generate_mock_trial_data
from pharma_trial_ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, build_preprocessor


def load_or_create_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_mock_trial_data().to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def candidate_models() -> list[tuple[str, Pipeline, dict]]:
    candidates = []

    logistic_params = [
        {"C": 0.1, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 3.0, "class_weight": "balanced"},
    ]

    for params in logistic_params:
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        C=params["C"],
                        class_weight=params["class_weight"],
                    ),
                ),
            ]
        )
        candidates.append(("logistic_regression", model, params))

    rf_params = [
        {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 10},
        {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 8},
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5},
    ]

    for params in rf_params:
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        candidates.append(("random_forest", model, params))

    return candidates


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_or_create_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    best_model = None
    best_metrics = None
    best_run_id = None
    best_score = -1.0

    for model_name, model, params in candidate_models():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            input_example = X_test.head(3)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
            )

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = model
                best_metrics = metrics
                best_run_id = run.info.run_id

    if best_model is None:
        raise RuntimeError("No model was trained.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "model_path": str(MODEL_PATH),
        "best_run_id": best_run_id,
        "best_metrics": best_metrics,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    print("Best model saved")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
