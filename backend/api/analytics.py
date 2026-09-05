"""
RiskShield AI - Analytics Endpoints

  GET /analytics/summary  -- headline numbers for the dashboard overview page
  GET /analytics/metrics  -- full evaluation report (classification/operational/cost/ring metrics)
  GET /analytics/clusters -- graph-based abuse ring clusters
"""
from fastapi import APIRouter

from backend.state import state

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def dashboard_summary():
    """Headline numbers for the merchant dashboard overview page."""
    m = state["metrics"]
    return {
        "transactions_analyzed": m["temporal_split"]["test_txns"],
        "fraud_detected": m["operational_metrics"]["true_positives"],
        "false_positives": m["operational_metrics"]["false_positives"],
        "precision": m["classification_metrics"]["precision"],
        "recall": m["classification_metrics"]["recall"],
        "pr_auc": m["classification_metrics"]["pr_auc"],
        "expected_fraud_loss_inr": m["cost_analysis"]["expected_fn_cost_inr"],
        "expected_fp_cost_inr": m["cost_analysis"]["expected_fp_cost_inr"],
        "total_expected_cost_inr": m["cost_analysis"]["total_expected_cost_inr"],
        "cost_reduction_vs_ml_only_pct": m["cost_reduction_vs_ml_only_baseline_pct"],
        "clusters_found": m["ring_detection_metrics"]["clusters_found"],
    }


@router.get("/metrics")
def dashboard_metrics():
    return state["metrics"]


@router.get("/clusters")
def list_clusters(limit: int = 50):
    df = state["clusters_df"].sort_values("cluster_size", ascending=False).head(limit)
    return df.to_dict(orient="records")
