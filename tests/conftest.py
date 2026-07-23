"""Shared test fixtures: a loaded model and a legit-looking sample transaction."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.service.app import app
from src.service.model import get_model


@pytest.fixture(scope="session")
def model():
    return get_model()


@pytest.fixture
def sample_transaction(model) -> dict:
    """A transaction with all features present; V* at zero, plausible Time/Amount."""
    txn = {c: 0.0 for c in model.feature_columns}
    txn["Time"] = 40000.0
    txn["Amount"] = 149.62
    return txn


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
