"""Data-drift monitoring: compare live scored traffic against the training baseline
and emit an Evidently HTML report.

The service logs every scored transaction to a JSONL file (see
`src/service/monitor_log.py`). This job reads that log as the "current" window and
compares its feature distributions to a "reference" sample of the training data,
flagging drift that would warrant retraining.

Runs in its own environment (Evidently pins older numpy/scikit-learn):

    pip install -r requirements-monitoring.txt
    python -m src.monitoring.drift --reference data/creditcard.csv \
        --current /tmp/predictions.jsonl --out docs/monitoring/drift_report.html

Designed to be scheduled (e.g. a nightly GitHub Actions cron).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# The 30 model features; Class/score/decision/ts are excluded from the drift check.
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_current(path: Path) -> pd.DataFrame:
    """Read the service's JSONL prediction log into a feature frame."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"No predictions found in {path}; exercise the API first.")
    return pd.DataFrame(records)[FEATURE_COLUMNS]


def load_reference(path: Path, n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Sample the training data as the drift baseline."""
    df = pd.read_csv(path)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df[FEATURE_COLUMNS]


def build_report(reference: pd.DataFrame, current: pd.DataFrame, out: Path) -> dict:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))

    result = report.as_dict()
    summary = result["metrics"][0]["result"]
    return {
        "n_drifted": summary.get("number_of_drifted_columns"),
        "share_drifted": summary.get("share_of_drifted_columns"),
        "dataset_drift": summary.get("dataset_drift"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=Path("data/creditcard.csv"))
    ap.add_argument("--current", type=Path, default=Path("/tmp/predictions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("docs/monitoring/drift_report.html"))
    ap.add_argument("--fail-on-drift", action="store_true", help="exit 1 if dataset drift is detected")
    args = ap.parse_args()

    reference = load_reference(args.reference)
    current = load_current(args.current)
    summary = build_report(reference, current, args.out)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    if args.fail_on_drift and summary.get("dataset_drift"):
        print("Dataset drift detected — retraining recommended.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
