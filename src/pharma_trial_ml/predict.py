from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from pharma_trial_ml.config import MODEL_PATH
from pharma_trial_ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m pharma_trial_ml.train"
        )
    return joblib.load(MODEL_PATH)


def predict_response(record: dict) -> dict:
    model = load_model()

    expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = pd.DataFrame([record], columns=expected_columns)

    probability = float(model.predict_proba(df)[0, 1])
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "responder_probability": round(probability, 4),
    }
