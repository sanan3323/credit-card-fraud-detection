# Credit Card Fraud Detection — Ops-Team Playbook

**Audience.** Fraud operations, model risk, and whoever owns the decline-decision engine. This is not a data-science writeup.

---

## What this model does, in two sentences

It scores every card transaction with a probability of being fraud based on 28 anonymized behavioral features plus `Amount` and `Time`. It emits an action — `allow`, `review`, or `block` — chosen to minimize **expected dollar loss** given a stated cost of missed fraud and a stated cost of false alarm.

## What this model does NOT do

- It does **not** verify identity.
- It does **not** predict legitimate customer chargeback disputes.
- It does **not** score merchants.
- It does **not** explain *why* in business language. SHAP shows which PCA components contributed, but those components are anonymized in the public dataset. Analysts with access to the unencrypted features can map them back.
- It is **not** fairness-audited on its current training data (Kaggle data exposes no demographics). Audit against protected attributes on the bank's internal data before go-live.

---

## How to use the score

Three-tier policy derived in the notebook:

| Score range                      | Action     | What this means operationally                 |
| -------------------------------- | ---------- | --------------------------------------------- |
| `score < t_review`               | **allow**  | Pass through. No friction.                    |
| `t_review ≤ score < t_block`     | **review** | Step-up auth or manual analyst review.        |
| `score ≥ t_block`                | **block**  | Auto-decline. High-precision tier.            |

The three thresholds (`t_review`, `t_star`, `t_block`) are saved inside `fraud_model_artifact.pkl` alongside the model and the scaler. **Do not hard-code thresholds elsewhere in the pipeline** — the threshold is part of the model. When you retrain, you get a new set.

### Inference code sketch

```python
import pickle, pandas as pd
bundle = pickle.load(open("fraud_model_artifact.pkl", "rb"))
model, scaler, thr = bundle["model"], bundle["scaler"], bundle["thresholds"]

def score_and_route(txn: pd.DataFrame) -> dict:
    x = txn[bundle["feature_columns"]].copy()
    x[["Time", "Amount"]] = scaler.transform(x[["Time", "Amount"]])
    p = float(model.predict_proba(x)[:, 1][0])
    if p >= thr["t_block"]:   action = "block"
    elif p >= thr["t_review"]: action = "review"
    else:                       action = "allow"
    return {"score": p, "action": action}
```

---

## Cost assumptions

Baseline cost matrix used to choose `t_star`:

- **`C_FN` (missed fraud)** — ~$122. This is the mean fraud transaction amount on training data. Adjust upward to account for chargeback processing (~$15–30) and reputational/churn cost. Production banks typically use `C_FN ≈ mean_amount × 1.3–1.5`.
- **`C_FP` (false alarm)** — $5. Estimate of analyst review time (~$1–2 loaded) plus customer friction when a legit transaction is declined or stepped-up.

**The threshold and the expected savings scale with these numbers.** If finance estimates `C_FP` at $25 instead of $5, rerun §6 of the notebook — the sensitivity plot there shows how much `t*` moves. Expect the threshold to drift up as `C_FP` rises (we become more reluctant to flag).

**Revisit the cost matrix quarterly** with fraud finance. The numbers above are starting points, not ground truth.

---

## Monitoring checklist (weekly)

- Score distribution vs training distribution — KS statistic < 0.05 ideal, investigate at > 0.10.
- Precision at block tier (from chargeback + confirmed-fraud feedback) — target ≥ 90%.
- Recall at review tier (from audits of confirmed frauds) — target ≥ 85%.
- Review queue volume vs analyst SLA — if queue grows faster than staffing, tighten `t_review`.
- False-decline complaint volume — stable week-over-week.

## Retraining triggers

Retrain from fresh labels when **any** of the following:

- PR-AUC on a rolling 4-week holdout drops > 5 percentage points.
- Precision at block tier falls below the 90% target for 2 consecutive weeks.
- A new attack pattern is reported by ops and a sample of confirmed cases scores below `t_review` (the model hasn't learned it).
- Quarterly schedule, regardless of the above.

Whenever you retrain, you must also rerun the cost-threshold section of the notebook and repackage the artifact. **Do not mix a new model with old thresholds.**

---

## Known limitations

- **Training data is two days of European transactions in September 2013.** Models destined for production must be retrained on the bank's own recent data. The Kaggle dataset exists here only to demonstrate the method.
- **V1–V28 are PCA components.** We can measure score drift but cannot attribute drift to a specific behavior without the pre-PCA features. Internal retraining on raw features will fix this.
- **Train/test split is stratified random, not temporal.** The two-day window makes a clean time split impractical for this public dataset. Before production deployment, validate on a time-based holdout (last N days) to rule out temporal leakage.
- **No fairness audit was performed** — see above.
- **No calibration check was performed.** Raw XGBoost probabilities are not guaranteed to be well-calibrated. If downstream systems interpret the score as a literal probability (e.g., for expected-loss accounting), add a Platt or isotonic calibration step on a held-out fold.

---

## Escalation

- Model behaving unexpectedly → data science on-call.
- Spike in false declines → product + data science jointly.
- Suspected novel attack not caught → fraud strategy, with a retrain request.

---

## File inventory

- `CreditCardFraud_refactored.ipynb` — the training notebook. Fully reproducible with `creditcard.csv` in the working directory.
- `fraud_model_artifact.pkl` — the deliverable. Contains the model, the scaler, the three thresholds, the cost matrix used, and training provenance.
- `README.md` — this file.

---

## For the data scientist reproducing this

1. Place `creditcard.csv` (Kaggle: `mlg-ulb/creditcardfraud`) next to the notebook.
2. Run all cells top to bottom. ~3–5 minutes on a modern laptop.
3. Review §6 with the ops team before publishing `t_star`. The default `C_FN` and `C_FP` are placeholders.
4. Commit `fraud_model_artifact.pkl` to your model registry alongside the training commit SHA and the `comparison` table printed in §5.1.
