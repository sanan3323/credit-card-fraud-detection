"""End-to-end API contract tests against the real loaded model."""
from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"]
    assert {"t_review", "t_star", "t_block"} <= set(body["thresholds"])


def test_predict_shape(client, sample_transaction):
    r = client.post("/predict", json=sample_transaction)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"block", "review", "allow"}
    assert 0.0 <= body["score"] <= 1.0
    assert len(body["top_features"]) == 5
    assert {"feature", "shap_value", "value"} <= set(body["top_features"][0])


def test_predict_missing_feature_is_rejected(client, sample_transaction):
    del sample_transaction["V14"]
    r = client.post("/predict", json=sample_transaction)
    assert r.status_code == 422  # pydantic validation error


def test_predict_negative_amount_is_rejected(client, sample_transaction):
    sample_transaction["Amount"] = -10.0
    r = client.post("/predict", json=sample_transaction)
    assert r.status_code == 422


def test_explanations_reference_real_features(client, sample_transaction, model):
    r = client.post("/predict", json=sample_transaction)
    feats = {f["feature"] for f in r.json()["top_features"]}
    assert feats <= set(model.feature_columns)
