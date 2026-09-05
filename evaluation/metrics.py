"""
RiskShield AI - Evaluation
Computes classification metrics, operational metrics, and financial
(cost-sensitive) metrics on a held-out temporal test set. Accuracy is
deliberately NOT the headline metric -- fraud is highly imbalanced, and
false positives/negatives have asymmetric costs.
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve, confusion_matrix
)

COST_FALSE_POSITIVE = 100   # INR: friction + lost legitimate revenue on a wrongly-held txn
COST_FALSE_NEGATIVE = 2000  # INR: average fraud loss per missed fraudulent txn


def classification_metrics(y_true, y_pred, y_score):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
    }


def operational_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)
    return {
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
        "alerts_per_1000_txns": round((tp + fp) / total * 1000, 2),
    }


def cost_analysis(y_true, y_pred, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fp_cost = fp * cost_fp
    fn_cost = fn * cost_fn
    return {
        "cost_per_false_positive": cost_fp,
        "cost_per_false_negative": cost_fn,
        "expected_fp_cost_inr": int(fp_cost),
        "expected_fn_cost_inr": int(fn_cost),
        "total_expected_cost_inr": int(fp_cost + fn_cost),
    }


def find_optimal_threshold(y_true, y_score, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE,
                            thresholds=None):
    """Sweep thresholds on the VALIDATION set to minimize total expected cost."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, float("inf")
    costs = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        c = cost_analysis(y_true, y_pred, cost_fp, cost_fn)["total_expected_cost_inr"]
        costs.append((t, c))
        if c < best_cost:
            best_cost = c
            best_t = t
    return best_t, best_cost, costs


def ring_detection_metrics(clusters_df, txns_with_labels):
    """Precision/recall of graph-based clusters against ground-truth fraud scenario labels."""
    if clusters_df is None or clusters_df.empty:
        return {"clusters_found": 0, "cluster_precision": None, "cluster_recall": None}

    total_ring_fraud_customers = set(
        txns_with_labels[txns_with_labels["scenario"] == "abuse_ring"]["customer_id"]
    )
    flagged_customers = set()
    true_positive_clusters = 0
    for _, row in clusters_df.iterrows():
        custs = set(row["customers"])
        flagged_customers |= custs
        if row["cluster_fraud_rate"] > 0.3:
            true_positive_clusters += 1

    precision = true_positive_clusters / len(clusters_df) if len(clusters_df) else 0
    recall = len(flagged_customers & total_ring_fraud_customers) / max(len(total_ring_fraud_customers), 1)

    return {
        "clusters_found": int(len(clusters_df)),
        "cluster_precision": round(precision, 4),
        "cluster_recall": round(recall, 4),
        "avg_cluster_size": round(clusters_df["cluster_size"].mean(), 2),
    }
