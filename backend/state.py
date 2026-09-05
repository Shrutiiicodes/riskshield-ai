"""
RiskShield AI - Shared Application State

Loaded once at FastAPI startup (see backend/main.py) and read by every
route module under backend/api/. Kept as a plain module-level dict rather
than a class/singleton because it's just cached read-only artifacts from
the last pipeline run -- no need for anything fancier.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

REPORT_DIR = Path(__file__).parent.parent / "evaluation" / "reports"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

state = {}


def load_artifacts():
    state["metrics"] = json.loads((REPORT_DIR / "metrics.json").read_text())
    state["test_df"] = pd.read_csv(REPORT_DIR / "test_set_scored.csv")
    state["clusters_df"] = pd.read_csv(REPORT_DIR / "clusters.csv")
    state["feature_cols"] = json.loads((REPORT_DIR / "feature_cols.json").read_text())
    state["clf"] = joblib.load(REPORT_DIR / "fraud_classifier.joblib")
    state["anom"] = joblib.load(REPORT_DIR / "anomaly_detector.joblib")
    return state
