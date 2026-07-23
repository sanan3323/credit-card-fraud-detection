"""Reproducible training pipeline for the fraud scoring model.

Turns the exploratory notebook into a rerunnable script: load data -> stratified
split -> scale Time/Amount -> fit five models -> pick the best by PR-AUC ->
derive the cost-based thresholds -> write the artifact + metrics.json.

Data: the Kaggle "Credit Card Fraud Detection" dataset (creditcard.csv, ~150 MB,
284,807 rows). It is gitignored (too large / redistribution terms). Download from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and pass its path:

    python -m src.train --data data/creditcard.csv

Outputs (default): artifacts/fraud_model_artifact.pkl and artifacts/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

try:  # optional; only the class_weight model needs no SMOTE, but we mirror the notebook
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False

SEED = 42
SCALED_COLUMNS = ["Time", "Amount"]
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="aucpr",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)


def load_split(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, RobustScaler]:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Class"]).copy()
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    scaler = RobustScaler()
    X_train[SCALED_COLUMNS] = scaler.fit_transform(X_train[SCALED_COLUMNS])
    X_test[SCALED_COLUMNS] = scaler.transform(X_test[SCALED_COLUMNS])
    return X_train, X_test, y_train, y_test, scaler


def fit_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=SEED
        ),
        "XGBoost": xgb.XGBClassifier(**XGB_PARAMS),
        "XGBoost+class_weight": xgb.XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}),
    }
    if _HAS_IMBLEARN:
        models["XGBoost+SMOTE"] = ImbPipeline(
            [("smote", SMOTE(random_state=SEED)), ("xgb", xgb.XGBClassifier(**XGB_PARAMS))]
        )
    return {name: m.fit(X_train, y_train) for name, m in models.items()}


def evaluate(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[str, object, dict, np.ndarray]:
    rows = {}
    for name, m in models.items():
        p = m.predict_proba(X_test)[:, 1]
        pred05 = (p >= 0.5).astype(int)
        rows[name] = {
            "PR-AUC": average_precision_score(y_test, p),
            "ROC-AUC": roc_auc_score(y_test, p),
            "Precision@0.5": precision_score(y_test, pred05, zero_division=0),
            "Recall@0.5": recall_score(y_test, pred05, zero_division=0),
            "F1@0.5": f1_score(y_test, pred05, zero_division=0),
        }
    best_name = max(rows, key=lambda n: rows[n]["PR-AUC"])
    return best_name, models[best_name], rows, models[best_name].predict_proba(X_test)[:, 1]


def derive_thresholds(y_test: pd.Series, proba: np.ndarray, c_fn: float, c_fp: float = 5.0) -> dict:
    grid = np.linspace(0.001, 0.999, 400)
    fn = np.array([((proba < t) & (y_test == 1)).sum() for t in grid])
    fp = np.array([((proba >= t) & (y_test == 0)).sum() for t in grid])
    costs = fn * c_fn + fp * c_fp
    t_star = float(grid[int(np.argmin(costs))])

    prec, rec, thr = precision_recall_curve(y_test, proba)
    block = np.where(prec[:-1] >= 0.90)[0]
    review = np.where(rec[:-1] >= 0.90)[0]
    t_block = float(thr[block[0]]) if len(block) else 1.0
    t_review = float(thr[review[-1]]) if len(review) else 0.0
    t_review = min(t_review, t_star, t_block)
    t_block = max(t_block, t_star)
    return {"t_review": t_review, "t_star": t_star, "t_block": t_block}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/fraud_model_artifact.pkl"))
    ap.add_argument("--metrics-out", type=Path, default=Path("artifacts/metrics.json"))
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Dataset not found at {args.data}. Download creditcard.csv from Kaggle "
            "(mlg-ulb/creditcardfraud) and pass --data <path>."
        )

    X_train, X_test, y_train, y_test, scaler = load_split(args.data)
    models = fit_models(X_train, y_train)
    best_name, best_model, comparison, proba = evaluate(models, X_test, y_test)

    # C_FN = mean fraud amount in dollars, computed on the original (pre-scaled) data.
    raw = pd.read_csv(args.data)
    c_fn = float(raw.loc[raw["Class"] == 1, "Amount"].mean())
    thresholds = derive_thresholds(y_test, proba, c_fn=c_fn)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "feature_columns": list(X_train.columns),
        "thresholds": thresholds,
        "cost_matrix": {"C_FN": c_fn, "C_FP": 5.0},
        "metrics": {k: float(v) for k, v in comparison[best_name].items()},
        "trained_on": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "pos_rate": float(y_train.mean()),
        },
        "notes": "Threshold is part of the model. Do not hard-code thresholds downstream.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(artifact, f)
    args.metrics_out.write_text(
        json.dumps(
            {"model": best_name, "thresholds": thresholds, "metrics": artifact["metrics"], "comparison": comparison},
            indent=2,
        )
    )
    print(f"Best model: {best_name}  PR-AUC={comparison[best_name]['PR-AUC']:.4f}")
    print(f"Wrote {args.out} and {args.metrics_out}")


if __name__ == "__main__":
    main()
