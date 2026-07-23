"""Smoke tests for the drift monitor's data plumbing.

The Evidently report itself runs in a separate environment (requirements-
monitoring.txt), so here we only exercise the log parsing that feeds it.
"""
from __future__ import annotations

import json

import pytest

from src.monitoring.drift import FEATURE_COLUMNS, load_current


def test_load_current_parses_prediction_log(tmp_path, model):
    log = tmp_path / "predictions.jsonl"
    txn = {c: 0.0 for c in model.feature_columns}
    txn.update({"Time": 40000.0, "Amount": 149.62})
    record = {"ts": "2026-07-23T00:00:00Z", "score": 0.01, "decision": "allow", **txn}
    log.write_text(json.dumps(record) + "\n")

    df = load_current(log)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_load_current_empty_log_errors(tmp_path):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    with pytest.raises(SystemExit):
        load_current(log)
