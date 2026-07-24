"""Shared model logic: load the artifact once, score a transaction, apply the
cost-based two-tier policy, and explain the decision with SHAP.

This is the single source of truth reused by the API (`app.py`), the training
script (`train.py`), and the tests. The thresholds live *inside* the artifact —
never hard-code them downstream (see the artifact's own `notes` field).
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# The artifact was pickled with a slightly older scikit-learn; the version banner
# is noise, not a correctness problem (we pin a compatible range in requirements).
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

_DEFAULT_ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "fraud_model_artifact.pkl"
ARTIFACT_PATH = Path(os.getenv("FRAUD_ARTIFACT_PATH", str(_DEFAULT_ARTIFACT)))

# Only these two raw columns were scaled at train time; V1..V28 are already PCA outputs.
_SCALED_COLUMNS = ["Time", "Amount"]


@dataclass(frozen=True)
class Decision:
    decision: str  # "block" | "review" | "allow"
    score: float
    threshold_hit: str  # which threshold gated the decision
    top_features: list[dict]  # [{feature, shap_value, value}, ...]


class FraudModel:
    """Loads the artifact and turns a raw transaction into an explained decision."""

    def __init__(self, artifact: dict):
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_columns: list[str] = artifact["feature_columns"]
        self.thresholds: dict = artifact["thresholds"]
        self.cost_matrix: dict = artifact["cost_matrix"]
        self.metrics: dict = artifact["metrics"]
        self.model_name: str = artifact.get("model_name", "unknown")
        self._explainer = None  # lazily built; SHAP TreeExplainer construction is cheap but deferred

    @classmethod
    def from_path(cls, path: Path | str = ARTIFACT_PATH) -> FraudModel:
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def _prepare(self, transaction: dict) -> pd.DataFrame:
        """Order columns as the model expects and apply the fitted scaler to Time/Amount."""
        missing = [c for c in self.feature_columns if c not in transaction]
        if missing:
            raise ValueError(f"missing feature(s): {missing}")
        row = pd.DataFrame([{c: transaction[c] for c in self.feature_columns}], columns=self.feature_columns)
        row[_SCALED_COLUMNS] = self.scaler.transform(row[_SCALED_COLUMNS])
        return row

    def score(self, transaction: dict) -> float:
        row = self._prepare(transaction)
        return float(self.model.predict_proba(row)[:, 1][0])

    def decide(self, transaction: dict, explain: bool = True, top_k: int = 5) -> Decision:
        row = self._prepare(transaction)
        score = float(self.model.predict_proba(row)[:, 1][0])
        decision, hit = classify(score, self.thresholds)
        top = self._explain(row, top_k) if explain else []
        return Decision(decision=decision, score=score, threshold_hit=hit, top_features=top)

    def _explain(self, row: pd.DataFrame, top_k: int) -> list[dict]:
        import shap  # imported lazily so non-serving callers don't pay the import cost

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        values = self._explainer.shap_values(row)
        vals = values[0] if isinstance(values, list) else values
        vals = np.asarray(vals).reshape(-1)
        order = np.argsort(np.abs(vals))[::-1][:top_k]
        return [
            {
                "feature": self.feature_columns[i],
                "shap_value": round(float(vals[i]), 5),
                "value": round(float(row.iloc[0, i]), 5),
            }
            for i in order
        ]


def classify(score: float, thresholds: dict) -> tuple[str, str]:
    """Cost-based two-tier policy. Order matters: block dominates review dominates allow."""
    if score >= thresholds["t_block"]:
        return "block", "t_block"
    if score >= thresholds["t_review"]:
        return "review", "t_review"
    return "allow", "below_t_review"


@lru_cache(maxsize=1)
def get_model() -> FraudModel:
    """Process-wide singleton so the artifact + SHAP explainer load exactly once."""
    return FraudModel.from_path()
