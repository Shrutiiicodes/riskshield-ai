"""
RiskShield AI - ML Models
  Model 1: Supervised fraud classifier (XGBoost)
  Model 2: Unsupervised anomaly detector (Isolation Forest)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class FraudClassifier:
    def __init__(self, feature_cols, params=None):
        self.feature_cols = feature_cols
        self.params = params or dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="aucpr",
            random_state=42,
        )
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # handle class imbalance with scale_pos_weight
        n_pos = max(y_train.sum(), 1)
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos
        self.model = xgb.XGBClassifier(**self.params, scale_pos_weight=scale_pos_weight)
        eval_set = [(X_val[self.feature_cols], y_val)] if X_val is not None else None
        self.model.fit(X_train[self.feature_cols], y_train, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.feature_cols])[:, 1]


class AnomalyDetector:
    def __init__(self, feature_cols, contamination=0.05):
        self.feature_cols = feature_cols
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
        )

    def fit(self, X_train):
        Xs = self.scaler.fit_transform(X_train[self.feature_cols])
        self.model.fit(Xs)
        return self

    def score(self, X):
        """Return anomaly score in [0, 1], higher = more anomalous."""
        Xs = self.scaler.transform(X[self.feature_cols])
        raw = self.model.score_samples(Xs)  # higher = more normal
        # normalize: invert & min-max scale to [0,1]
        norm = (raw.max() - raw) / (raw.max() - raw.min() + 1e-9)
        return norm


def fuse_risk(ml_score, anomaly_score, network_score, weights=(0.6, 0.2, 0.2)):
    w_ml, w_anom, w_net = weights
    return np.clip(w_ml * ml_score + w_anom * anomaly_score + w_net * network_score, 0, 1)


def risk_level(score):
    if score < 0.30:
        return "LOW"
    elif score < 0.60:
        return "MEDIUM"
    elif score < 0.85:
        return "HIGH"
    return "CRITICAL"
