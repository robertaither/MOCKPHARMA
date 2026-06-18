import pytest

from pharma_trial_ml.data import generate_mock_trial_data
from pharma_trial_ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def test_model_input_contract_columns():
    df = generate_mock_trial_data(n_rows=10)
    model_input = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    assert list(model_input.columns) == [
        "age",
        "bmi",
        "baseline_biomarker",
        "dose_mg",
        "adherence_pct",
        "prior_treatments",
        "sex",
        "trial_arm",
    ]
