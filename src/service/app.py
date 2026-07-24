"""FastAPI service exposing the fraud scoring model.

    POST /predict  -> cost-based block/review/allow decision + SHAP explanation
    GET  /health   -> liveness + which model/thresholds are loaded

Run locally:  uvicorn src.service.app:app --reload
The same app is wrapped by Mangum for AWS Lambda (see `handler` at the bottom).
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .model import get_model
from .monitor_log import log_prediction
from .schemas import HealthResponse, PredictionResponse, Transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-service")

app = FastAPI(
    title="Fraud Scoring Service",
    version="1.0.0",
    description="Cost-based credit-card fraud scoring with SHAP explanations.",
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the bare URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = get_model()
    return HealthResponse(status="ok", model_name=model.model_name, thresholds=model.thresholds)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    model = get_model()
    result = model.decide(transaction.model_dump())
    log_prediction(transaction.model_dump(), result)
    return PredictionResponse(
        decision=result.decision,
        score=result.score,
        threshold_hit=result.threshold_hit,
        top_features=result.top_features,
    )


try:  # only needed in the Lambda image; absent locally is fine
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:  # pragma: no cover
    handler = None
