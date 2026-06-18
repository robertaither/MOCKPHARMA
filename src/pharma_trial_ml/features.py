from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "bmi",
    "baseline_biomarker",
    "dose_mg",
    "adherence_pct",
    "prior_treatments",
]

CATEGORICAL_FEATURES = [
    "sex",
    "trial_arm",
]

TARGET = "responder"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
