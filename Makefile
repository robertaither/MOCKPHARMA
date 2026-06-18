.PHONY: install data train api test docker-build docker-run mlflow-ui

install:
	pip install -r requirements.txt

data:
	python -m pharma_trial_ml.data

train:
	python -m pharma_trial_ml.train

api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

docker-build:
	docker build -t mock-pharma-mlflow-api .

docker-run:
	docker run -p 8000:8000 mock-pharma-mlflow-api
