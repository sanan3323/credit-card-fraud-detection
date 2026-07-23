"""Request/response schemas for the fraud scoring API.

A transaction is the 30 raw features from the source data: Time, V1..V28, Amount.
V1..V28 are anonymised PCA components; Time and Amount are raw and get scaled
server-side by the model's own fitted scaler.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, create_model

_V_FIELDS = {f"V{i}": (float, Field(..., description=f"PCA component V{i}")) for i in range(1, 29)}

Transaction = create_model(
    "Transaction",
    Time=(float, Field(..., description="Seconds elapsed since the first transaction in the dataset")),
    Amount=(float, Field(..., ge=0, description="Transaction amount")),
    **_V_FIELDS,
    __doc__="One credit-card transaction: Time, V1..V28, Amount.",
)


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    value: float


class PredictionResponse(BaseModel):
    decision: str = Field(..., description="block | review | allow")
    score: float = Field(..., description="Predicted fraud probability in [0, 1]")
    threshold_hit: str = Field(..., description="Which policy threshold gated the decision")
    top_features: list[ShapFeature] = Field(..., description="Most influential features (SHAP)")


class HealthResponse(BaseModel):
    status: str
    model_name: str
    thresholds: dict
