# Mock Pharma Trial MLflow Production Project

This is an end-to-end machine learning project for a mock pharmaceutical clinical-trial use case.

The project demonstrates:

- synthetic clinical-trial data generation
- model training and experimentation
- hyperparameter search
- MLflow experiment tracking
- model artifact logging
- production-style model loading
- FastAPI prediction endpoint
- Docker deployment
- pytest tests
- GitHub Actions CI

The example task is binary classification:

> Predict whether a trial participant is likely to respond to treatment.

This is portfolio/demo code only. It does not use real patient data and must not be used for medical decision-making.

---

## Project structure

```text
mock-pharma-mlflow-production/
├── api/
│   └── app.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   └── pharma_trial_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── train.py
│       └── predict.py
├── tests/
│   ├── test_data.py
│   └── test_prediction_contract.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

## 1. Create environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Generate mock clinical-trial data

```bash
python -m pharma_trial_ml.data
```

This creates:

```text
data/mock_trial_data.csv
```

---

## 3. Train the model with MLflow tracking

```bash
python -m pharma_trial_ml.train
```

This will:

- train several candidate models
- log parameters, metrics and artifacts to MLflow
- save the best model locally
- create an MLflow experiment called `mock-pharma-trial-response`

---

## 4. Open MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

You can compare runs, metrics, parameters and artifacts.

---

## 5. Run the prediction API locally

Train the model first, then run:

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "age": 54,
  "bmi": 29.5,
  "baseline_biomarker": 68.2,
  "dose_mg": 50,
  "adherence_pct": 92,
  "prior_treatments": 1,
  "sex": "F",
  "trial_arm": "active"
}
```

---

## 6. Run with Docker

Build:

```bash
docker build -t mock-pharma-mlflow-api .
```

Run:

```bash
docker run -p 8000:8000 mock-pharma-mlflow-api
```

Then visit:

```text
http://127.0.0.1:8000/docs
```

---

## 7. Run tests

```bash
pytest
```

---

## Portfolio explanation

This project shows a production-style ML workflow:

1. Data is generated and versioned locally.
2. Training runs are logged to MLflow.
3. Metrics and parameters are comparable across runs.
4. The best model is saved as a production artifact.
5. The API loads the model artifact rather than retraining.
6. Docker makes the service deployable.

---

## Possible extensions
# MOCKPHARMA
# MOCKPHARMA
