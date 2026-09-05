"""
RiskShield AI - End-to-End Pipeline

  1. Load synthetic data
  2. Feature engineering
  3. Temporal split: Jan-Aug train / Sep validation / Oct held-out test
  4. Train fraud classifier (XGBoost) + anomaly detector (Isolation Forest)
  5. Graph-based abuse ring detection (NetworkX)
  6. Risk fusion + threshold tuning on VALIDATION set only
  7. Evaluate on held-out TEST set (metrics, cost analysis, ring detection)
  8. SHAP explainability for sample high-risk transactions
  9. Run the investigation agent on top alerts
  10. Save all results to evaluation/reports/

This script produces REAL measured numbers -- nothing here is invented.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

sys.path.insert(0, str(Path(__file__).parent))

from backend.features.engineering import load_raw, build_features
from backend.models.fraud_models import FraudClassifier, AnomalyDetector, fuse_risk, risk_level
from backend.graph.ring_detector import detect_rings
from backend.policies.risk_policy import evaluate_policy
from backend.agents.investigation_agent import investigate, build_evidence_reasons
from evaluation.metrics import (
    classification_metrics, operational_metrics, cost_analysis,
    find_optimal_threshold, ring_detection_metrics,
)

DATA_DIR = Path(__file__).parent / "data" / "processed"
REPORT_DIR = Path(__file__).parent / "evaluation" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 120)


def main():
    print("=" * 70)
    print("RISKSHIELD AI - PIPELINE RUN")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1-2. Load + feature engineering
    # -----------------------------------------------------------------
    print("\n[1/8] Loading data & engineering features...")
    txns, customers, devices, ips, merchants = load_raw(DATA_DIR)
    scenario_labels = pd.read_csv(DATA_DIR / "transactions_scenario_labels.csv")
    txns = txns.merge(scenario_labels, on="transaction_id", how="left")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], format="mixed")

    df, feature_cols = build_features(txns, customers, devices, ips, merchants)
    print(f"  {len(df):,} transactions, {len(feature_cols)} features")
    print(f"  Features: {feature_cols}")

    # -----------------------------------------------------------------
    # 3. Temporal split
    # -----------------------------------------------------------------
    print("\n[2/8] Temporal split (Jan-Aug train / Sep val / Oct test)...")
    train_mask = df["timestamp"] < "2025-09-01"
    val_mask = (df["timestamp"] >= "2025-09-01") & (df["timestamp"] < "2025-10-01")
    test_mask = df["timestamp"] >= "2025-10-01"

    train_df, val_df, test_df = df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()
    print(f"  Train: {len(train_df):,} ({train_df.label.mean()*100:.2f}% fraud)")
    print(f"  Val:   {len(val_df):,} ({val_df.label.mean()*100:.2f}% fraud)")
    print(f"  Test:  {len(test_df):,} ({test_df.label.mean()*100:.2f}% fraud)")

    for c in feature_cols:
        for d in (train_df, val_df, test_df):
            d[c] = d[c].fillna(0)

    # -----------------------------------------------------------------
    # 4. Train models
    # -----------------------------------------------------------------
    print("\n[3/8] Training fraud classifier (XGBoost)...")
    clf = FraudClassifier(feature_cols).fit(train_df, train_df["label"], val_df, val_df["label"])

    print("[3/8] Training anomaly detector (Isolation Forest)...")
    anom = AnomalyDetector(feature_cols, contamination=0.05).fit(train_df)

    for split_df in (train_df, val_df, test_df):
        split_df["ml_score"] = clf.predict_proba(split_df)
        split_df["anomaly_score"] = anom.score(split_df)

    # -----------------------------------------------------------------
    # 5. Graph-based ring detection (fit on full history up to test end,
    #    since network structure should be visible cumulatively)
    # -----------------------------------------------------------------
    print("\n[4/8] Running graph-based abuse ring detection...")
    all_scored, clusters_df = detect_rings(df, min_customers=3)
    df = df.merge(
        all_scored[["transaction_id", "network_risk_score", "cluster_id", "cluster_size"]],
        on="transaction_id", how="left"
    )
    train_df = train_df.merge(all_scored[["transaction_id", "network_risk_score"]], on="transaction_id", how="left")
    val_df = val_df.merge(all_scored[["transaction_id", "network_risk_score"]], on="transaction_id", how="left")
    test_df = test_df.merge(all_scored[["transaction_id", "network_risk_score", "cluster_id", "cluster_size"]], on="transaction_id", how="left")
    print(f"  {len(clusters_df)} suspicious clusters found (>= 3 linked customers)")

    # -----------------------------------------------------------------
    # 6. Risk fusion + threshold tuning on VALIDATION set
    # -----------------------------------------------------------------
    print("\n[5/8] Fusing risk scores & tuning threshold on validation set...")
    weights = (0.6, 0.2, 0.2)
    val_df["risk_score"] = fuse_risk(val_df["ml_score"], val_df["anomaly_score"], val_df["network_risk_score"], weights)
    test_df["risk_score"] = fuse_risk(test_df["ml_score"], test_df["anomaly_score"], test_df["network_risk_score"], weights)
    train_df["risk_score"] = fuse_risk(train_df["ml_score"], train_df["anomaly_score"], train_df["network_risk_score"], weights)

    best_t, best_val_cost, cost_sweep = find_optimal_threshold(val_df["label"], val_df["risk_score"])
    print(f"  Optimal threshold (min expected cost on val set): {best_t:.2f}  (val cost ₹{best_val_cost:,})")

    # -----------------------------------------------------------------
    # 7. Evaluate on held-out TEST set
    # -----------------------------------------------------------------
    print("\n[6/8] Evaluating on held-out TEST set...")
    test_df["risk_level"] = test_df["risk_score"].apply(risk_level)
    test_df["y_pred"] = (test_df["risk_score"] >= best_t).astype(int)

    clf_metrics = classification_metrics(test_df["label"], test_df["y_pred"], test_df["risk_score"])
    op_metrics = operational_metrics(test_df["label"], test_df["y_pred"])
    cost_metrics = cost_analysis(test_df["label"], test_df["y_pred"])

    # baseline comparison: ML-only score (no fusion) at its own cost-optimal threshold on val
    val_ml_t, _, _ = find_optimal_threshold(val_df["label"], val_df["ml_score"])
    test_pred_ml_only = (test_df["ml_score"] >= val_ml_t).astype(int)
    baseline_cost = cost_analysis(test_df["label"], test_pred_ml_only)
    cost_reduction_pct = round(
        100 * (baseline_cost["total_expected_cost_inr"] - cost_metrics["total_expected_cost_inr"])
        / max(baseline_cost["total_expected_cost_inr"], 1), 2
    )

    test_clusters = clusters_df.copy()
    ring_metrics = ring_detection_metrics(test_clusters, df)

    print(f"  Precision: {clf_metrics['precision']:.3f}  Recall: {clf_metrics['recall']:.3f}  "
          f"F1: {clf_metrics['f1']:.3f}  PR-AUC: {clf_metrics['pr_auc']:.3f}")
    print(f"  Expected total cost (fused model): ₹{cost_metrics['total_expected_cost_inr']:,}")
    print(f"  Expected total cost (ML-only baseline): ₹{baseline_cost['total_expected_cost_inr']:,}")
    print(f"  Cost reduction vs ML-only baseline: {cost_reduction_pct}%")

    # -----------------------------------------------------------------
    # 8. SHAP explainability
    # -----------------------------------------------------------------
    print("\n[7/8] Computing SHAP explanations for top alerts...")
    explainer = shap.TreeExplainer(clf.model)
    top_alerts = test_df[test_df["risk_level"].isin(["HIGH", "CRITICAL"])].sort_values(
        "risk_score", ascending=False
    ).head(5)

    investigation_examples = []
    if len(top_alerts) > 0:
        shap_values = explainer.shap_values(top_alerts[feature_cols])
        for i, (idx, row) in enumerate(top_alerts.iterrows()):
            contrib = dict(zip(feature_cols, shap_values[i]))
            reasons = build_evidence_reasons(row, contrib)
            cluster_info = None
            if row.get("cluster_id", 0) and row["cluster_id"] > 0:
                crow = clusters_df[clusters_df["cluster_id"] == row["cluster_id"]]
                if not crow.empty:
                    cluster_info = {"cluster_size": int(crow.iloc[0]["cluster_size"]),
                                     "shared_device_count": int(crow.iloc[0]["shared_device_count"])}
            related_count = int((df["customer_id"] == row["customer_id"]).sum())

            evidence = {
                "transaction": {"transaction_id": row["transaction_id"], "amount": row["amount"],
                                 "customer_id": row["customer_id"], "merchant_id": row["merchant_id"]},
                "risk_score": float(row["risk_score"]),
                "risk_level": row["risk_level"],
                "reasons": reasons,
                "cluster": cluster_info,
                "related_transaction_count": related_count,
            }
            investigation = investigate(evidence)
            policy_decision = evaluate_policy(row["risk_level"], investigation["recommended_action"])

            investigation_examples.append({
                "transaction_id": row["transaction_id"],
                "amount": float(row["amount"]),
                "risk_score": float(row["risk_score"]),
                "risk_level": row["risk_level"],
                "ground_truth_label": int(row["label"]),
                "ground_truth_scenario": row.get("scenario", "unknown"),
                "evidence_reasons": reasons,
                "investigation_summary": investigation["summary"],
                "agent_recommended_action": investigation["recommended_action"],
                "agent_confidence": investigation["confidence"],
                "policy_final_action": policy_decision.action,
                "policy_reason": policy_decision.reason,
            })
            print(f"  [{row['risk_level']}] {row['transaction_id']}  score={row['risk_score']:.2f}  "
                  f"truth={'FRAUD' if row['label'] else 'legit'} ({row.get('scenario')})")

    # -----------------------------------------------------------------
    # 9. Save reports
    # -----------------------------------------------------------------
    print("\n[8/8] Saving evaluation reports & plots...")

    results = {
        "temporal_split": {
            "train_txns": int(len(train_df)), "val_txns": int(len(val_df)), "test_txns": int(len(test_df)),
            "train_fraud_rate": round(float(train_df.label.mean()), 4),
            "val_fraud_rate": round(float(val_df.label.mean()), 4),
            "test_fraud_rate": round(float(test_df.label.mean()), 4),
        },
        "fusion_weights": {"ml": weights[0], "anomaly": weights[1], "network": weights[2]},
        "optimal_threshold": round(float(best_t), 3),
        "classification_metrics": clf_metrics,
        "operational_metrics": op_metrics,
        "cost_analysis": cost_metrics,
        "baseline_ml_only_cost_analysis": baseline_cost,
        "cost_reduction_vs_ml_only_baseline_pct": cost_reduction_pct,
        "ring_detection_metrics": ring_metrics,
        "sample_investigations": investigation_examples,
    }
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plots
    from sklearn.metrics import precision_recall_curve, roc_curve
    prec, rec, _ = precision_recall_curve(test_df["label"], test_df["risk_score"])
    fpr, tpr, _ = roc_curve(test_df["label"], test_df["risk_score"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(rec, prec)
    axes[0, 0].set_title(f"Precision-Recall Curve (PR-AUC={clf_metrics['pr_auc']:.3f})")
    axes[0, 0].set_xlabel("Recall"); axes[0, 0].set_ylabel("Precision")

    axes[0, 1].plot(fpr, tpr)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set_title(f"ROC Curve (AUC={clf_metrics['roc_auc']:.3f})")
    axes[0, 1].set_xlabel("False Positive Rate"); axes[0, 1].set_ylabel("True Positive Rate")

    thresholds_arr = [c[0] for c in cost_sweep]
    costs_arr = [c[1] for c in cost_sweep]
    axes[1, 0].plot(thresholds_arr, costs_arr)
    axes[1, 0].axvline(best_t, color="red", linestyle="--", label=f"optimal t={best_t:.2f}")
    axes[1, 0].set_title("Expected Cost vs Threshold (validation set)")
    axes[1, 0].set_xlabel("Threshold"); axes[1, 0].set_ylabel("Expected Cost (INR)")
    axes[1, 0].legend()

    axes[1, 1].hist(test_df[test_df.label == 0]["risk_score"], bins=40, alpha=0.6, label="legit", density=True)
    axes[1, 1].hist(test_df[test_df.label == 1]["risk_score"], bins=40, alpha=0.6, label="fraud", density=True)
    axes[1, 1].set_title("Risk Score Distribution (Test Set)")
    axes[1, 1].set_xlabel("Fused Risk Score"); axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "evaluation_plots.png", dpi=130)
    plt.close()

    # feature importance plot
    importances = pd.Series(clf.model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=130)
    plt.close()

    # save scored test set + clusters for the dashboard / backend to use
    test_df.to_csv(REPORT_DIR / "test_set_scored.csv", index=False)
    clusters_df.to_csv(REPORT_DIR / "clusters.csv", index=False)
    df.to_csv(REPORT_DIR / "all_scored_transactions.csv", index=False)

    import joblib
    joblib.dump(clf, REPORT_DIR / "fraud_classifier.joblib")
    joblib.dump(anom, REPORT_DIR / "anomaly_detector.joblib")
    with open(REPORT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    print(f"\nSaved: {REPORT_DIR}/metrics.json, evaluation_plots.png, feature_importance.png, "
          f"test_set_scored.csv, clusters.csv, all_scored_transactions.csv, model artifacts")
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    return results


if __name__ == "__main__":
    main()
