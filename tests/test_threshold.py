"""The cost-based two-tier policy is the heart of the model's business value,
so pin its boundary behaviour exactly."""
from __future__ import annotations

import pytest

from src.service.model import classify

THRESHOLDS = {"t_review": 0.01, "t_star": 0.1, "t_block": 0.8}


@pytest.mark.parametrize(
    "score,expected_decision,expected_hit",
    [
        (0.90, "block", "t_block"),        # above t_block
        (0.80, "block", "t_block"),        # exactly t_block -> block (>= is inclusive)
        (0.50, "review", "t_review"),      # between review and block
        (0.01, "review", "t_review"),      # exactly t_review -> review
        (0.005, "allow", "below_t_review"),  # below t_review
        (0.0, "allow", "below_t_review"),  # zero
    ],
)
def test_classify_boundaries(score, expected_decision, expected_hit):
    decision, hit = classify(score, THRESHOLDS)
    assert decision == expected_decision
    assert hit == expected_hit


def test_block_dominates_review():
    """Ordering matters: a high score must never be downgraded to review."""
    decision, _ = classify(0.99, THRESHOLDS)
    assert decision == "block"


def test_real_model_thresholds_are_ordered(model):
    t = model.thresholds
    assert t["t_review"] <= t["t_star"] <= t["t_block"], t
