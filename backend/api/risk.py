"""
RiskShield AI - Risk Endpoints

  GET /risk/alerts               -- list flagged transactions, optionally by risk tier
  GET /risk/investigate/{txn_id} -- run the investigation agent on a flagged transaction
"""
import pandas as pd
import shap
from fastapi import APIRouter, HTTPException

from backend.state import state
from backend.agents.investigation_agent import investigate, build_evidence_reasons
from backend.policies.risk_policy import evaluate_policy

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/alerts")
def list_alerts(risk: str = None, limit: int = 50):
    """List flagged transactions from the held-out test set, optionally filtered by risk tier."""
    df = state["test_df"]
    if risk:
        df = df[df["risk_level"] == risk.upper()]
    df = df.sort_values("risk_score", ascending=False).head(limit)
    cols = ["transaction_id", "customer_id", "merchant_id", "amount", "risk_score",
            "risk_level", "label", "scenario"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict(orient="records")


@router.get("/investigate/{transaction_id}")
def investigate_transaction(transaction_id: str):
    """Run the investigation agent on a specific (already-scored) transaction."""
    df = state["test_df"]
    row = df[df["transaction_id"] == transaction_id]
    if row.empty:
        raise HTTPException(404, f"Transaction {transaction_id} not found in scored test set")
    row = row.iloc[0]

    explainer = shap.TreeExplainer(state["clf"].model)
    shap_vals = explainer.shap_values(pd.DataFrame([row[state["feature_cols"]]]))[0]
    contrib = dict(zip(state["feature_cols"], shap_vals))
    reasons = build_evidence_reasons(row, contrib)

    cluster_info = None
    if row.get("cluster_id", 0) and row["cluster_id"] > 0:
        crow = state["clusters_df"][state["clusters_df"]["cluster_id"] == row["cluster_id"]]
        if not crow.empty:
            cluster_info = {"cluster_size": int(crow.iloc[0]["cluster_size"]),
                             "shared_device_count": int(crow.iloc[0]["shared_device_count"])}

    evidence = {
        "transaction": {"transaction_id": row["transaction_id"], "amount": float(row["amount"]),
                         "customer_id": row["customer_id"], "merchant_id": row["merchant_id"]},
        "risk_score": float(row["risk_score"]),
        "risk_level": row["risk_level"],
        "reasons": reasons,
        "cluster": cluster_info,
        "related_transaction_count": int((df["customer_id"] == row["customer_id"]).sum()),
    }
    investigation = investigate(evidence)
    policy_decision = evaluate_policy(row["risk_level"], investigation["recommended_action"])

    return {
        "transaction": evidence["transaction"],
        "risk_score": evidence["risk_score"],
        "risk_level": evidence["risk_level"],
        "evidence_reasons": reasons,
        "cluster": cluster_info,
        "investigation_summary": investigation["summary"],
        "agent_recommended_action": investigation["recommended_action"],
        "agent_confidence": investigation["confidence"],
        "policy_final_action": policy_decision.action,
        "policy_allowed_agent_request": policy_decision.allowed,
        "policy_reason": policy_decision.reason,
        "ground_truth_label": int(row["label"]) if "label" in row else None,
        "ground_truth_scenario": row.get("scenario"),
    }
