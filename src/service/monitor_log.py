"""Append every scored transaction to a JSONL log so the drift monitor can later
compare production traffic against the training baseline.

Best-effort and non-blocking to the request: on a read-only filesystem (e.g. AWS
Lambda outside /tmp) logging is skipped rather than failing the prediction. Set
PREDICTION_LOG to relocate it (default /tmp so it works in Lambda too).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Decision

logger = logging.getLogger("fraud-service")

LOG_PATH = Path(os.getenv("PREDICTION_LOG", "/tmp/predictions.jsonl"))


def log_prediction(transaction: dict, result: "Decision") -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "score": result.score,
        "decision": result.decision,
        **transaction,
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:  # read-only FS or disk full — never break scoring over logging
        logger.warning("prediction log skipped: %s", e)
