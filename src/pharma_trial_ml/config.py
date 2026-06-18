from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATA_PATH = DATA_DIR / "mock_trial_data.csv"
MODEL_PATH = MODEL_DIR / "trial_response_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

EXPERIMENT_NAME = "mock-pharma-trial-response"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
