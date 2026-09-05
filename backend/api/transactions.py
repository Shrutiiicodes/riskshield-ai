"""
RiskShield AI - Transaction Endpoints

  POST /transactions/score   -- score an arbitrary transaction feature vector
  GET  /transactions/{id}    -- look up a scored transaction from the test set
"""
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.state import state
from backend.models.fraud_models import fuse_risk, risk_level
from backend.policies.risk_policy import evaluate_policy

router = APIRouter(prefix="/transactions", tags=["transactions"])


class LiveTransaction(BaseModel):
    amount: float
    amount_dev_ratio: float = 1.0
    amount_zscore_merchant: float = 0.0
    account_age_days: int = 365
    transaction_frequency_per_week: float = 1.0
    historical_chargebacks: int = 0
    historical_refunds: int = 0
    is_foreign_country: int = 0
    device_is_new: int = 0
    device_shared_flag: int = 0
    ip_shared_flag: int = 0
    device_customer_count: int = 1
    ip_customer_count: int = 1
    velocity_5min: int = 1
    velocity_1hr: int = 1
    seconds_since_prev_txn: float = 3600
    failed_txn_count_24h: int = 0
    historical_fraud_rate: float = 0.01


@router.post("/score")
def score_live_transaction(txn: LiveTransaction):
    """Score an arbitrary transaction feature vector without needing it in the test set."""
    row = pd.DataFrame([txn.model_dump()])
    ml_score = float(state["clf"].predict_proba(row)[0])
    anomaly_score = float(state["anom"].score(row)[0])
    # no network context for a synthetic live call -> assume 0 network risk
    fused = float(fuse_risk(pd.Series([ml_score]), pd.Series([anomaly_score]), pd.Series([0.0]))[0])
    level = risk_level(fused)
    policy_decision = evaluate_policy(level)
    return {
        "ml_score": round(ml_score, 4),
        "anomaly_score": round(anomaly_score, 4),
        "fused_risk_score": round(fused, 4),
        "risk_level": level,
        "policy_action": policy_decision.action,
    }


@router.get("/{transaction_id}")
def get_transaction(transaction_id: str):
    """Raw lookup of a single scored transaction from the held-out test set."""
    df = state["test_df"]
    row = df[df["transaction_id"] == transaction_id]
    if row.empty:
        raise HTTPException(404, f"Transaction {transaction_id} not found in scored test set")
    return row.iloc[0].to_dict()
