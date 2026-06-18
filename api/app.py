from fastapi import FastAPI
from pydantic import BaseModel, Field

from pharma_trial_ml.predict import predict_response


class TrialParticipant(BaseModel):
    age: float = Field(..., ge=18, le=100)
    bmi: float = Field(..., ge=10, le=60)
    baseline_biomarker: float = Field(..., ge=0, le=200)
    dose_mg: int = Field(..., ge=0, le=200)
    adherence_pct: float = Field(..., ge=0, le=100)
    prior_treatments: int = Field(..., ge=0, le=20)
    sex: str = Field(..., pattern="^(M|F)$")
    trial_arm: str = Field(..., pattern="^(placebo|active)$")


app = FastAPI(
    title="Mock Pharma Trial Response API",
    description="Production-style FastAPI service using an MLflow-tracked model artifact.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(participant: TrialParticipant) -> dict:
    return predict_response(participant.model_dump())
