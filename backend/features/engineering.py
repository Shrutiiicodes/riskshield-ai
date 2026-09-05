"""
RiskShield AI - Feature Engineering
Builds per-transaction features:
  - velocity (txn count in trailing windows per customer/device/ip)
  - amount deviation from customer baseline
  - device / ip novelty (first time seen, how many customers share it)
  - geographic deviation from home country
  - failure/chargeback/refund history
"""
import pandas as pd
import numpy as np


def load_raw(data_dir):
    txns = pd.read_csv(data_dir / "transactions.csv")
    txns["timestamp"] = pd.to_datetime(txns["timestamp"], format="mixed")
    customers = pd.read_csv(data_dir / "customers.csv")
    devices = pd.read_csv(data_dir / "devices.csv")
    devices["first_seen"] = pd.to_datetime(devices["first_seen"], format="mixed", errors="coerce")
    ips = pd.read_csv(data_dir / "ips.csv")
    merchants = pd.read_csv(data_dir / "merchants.csv")
    return txns, customers, devices, ips, merchants


def build_features(txns, customers, devices, ips, merchants):
    df = txns.sort_values(["customer_id", "timestamp"]).reset_index(drop=True).copy()

    # --- join static entity attributes ---
    df = df.merge(customers, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(devices[["device_id", "customer_count", "transaction_count", "first_seen"]]
                  .rename(columns={"customer_count": "device_customer_count",
                                    "transaction_count": "device_transaction_count",
                                    "first_seen": "device_first_seen"}),
                  on="device_id", how="left")
    df = df.merge(ips[["ip_id", "customer_count", "transaction_count"]]
                  .rename(columns={"customer_count": "ip_customer_count",
                                    "transaction_count": "ip_transaction_count"}),
                  on="ip_id", how="left")
    df = df.merge(merchants[["merchant_id", "historical_fraud_rate", "avg_transaction_value"]]
                  .rename(columns={"avg_transaction_value": "merchant_avg_txn_value"}),
                  on="merchant_id", how="left")

    # --- amount deviation ---
    df["amount_dev_ratio"] = df["amount"] / df["avg_transaction_amount"].replace(0, np.nan)
    df["amount_dev_ratio"] = df["amount_dev_ratio"].fillna(1.0)
    df["amount_zscore_merchant"] = (df["amount"] - df["merchant_avg_txn_value"]) / df["merchant_avg_txn_value"].replace(0, np.nan)
    df["amount_zscore_merchant"] = df["amount_zscore_merchant"].fillna(0.0)

    # --- geo deviation ---
    df["is_foreign_country"] = (df["country"] != df["home_country"]).astype(int)

    # --- device / ip novelty ---
    df["device_is_new"] = (df["device_first_seen"] == df["timestamp"]).astype(int)
    # fallback: if account age is small relative to device history missing
    df["device_shared_flag"] = (df["device_customer_count"] > 1).astype(int)
    df["ip_shared_flag"] = (df["ip_customer_count"] > 1).astype(int)

    # --- velocity features: trailing-window txn counts per customer ---
    df = df.sort_values(["customer_id", "timestamp"])
    df["prev_txn_time"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["seconds_since_prev_txn"] = (df["timestamp"] - df["prev_txn_time"]).dt.total_seconds()
    df["seconds_since_prev_txn"] = df["seconds_since_prev_txn"].fillna(999999)

    def rolling_count(group, window_minutes):
        s = group.set_index("timestamp")["transaction_id"]
        counts = []
        times = group["timestamp"].values
        # simple O(n^2) per-customer window count (customer txn counts are small enough)
        for i, t in enumerate(times):
            lo = t - np.timedelta64(window_minutes, "m")
            counts.append(int(((times <= t) & (times > lo)).sum()))
        return counts

    velocity_5min = []
    velocity_1hr = []
    for cid, group in df.groupby("customer_id", sort=False):
        velocity_5min.extend(rolling_count(group, 5))
        velocity_1hr.extend(rolling_count(group, 60))
    df["velocity_5min"] = velocity_5min
    df["velocity_1hr"] = velocity_1hr

    # --- failed transaction trailing count (last 24h) per customer ---
    df["is_failed"] = (df["transaction_status"] == "failed").astype(int)
    fail_counts = []
    for cid, group in df.groupby("customer_id", sort=False):
        times = group["timestamp"].values
        fails = group["is_failed"].values
        counts = []
        for i, t in enumerate(times):
            lo = t - np.timedelta64(24, "h")
            mask = (times <= t) & (times > lo)
            counts.append(int(fails[mask].sum()))
        fail_counts.extend(counts)
    df["failed_txn_count_24h"] = fail_counts

    # --- final feature set ---
    feature_cols = [
        "amount", "amount_dev_ratio", "amount_zscore_merchant",
        "account_age_days", "transaction_frequency_per_week",
        "historical_chargebacks", "historical_refunds",
        "is_foreign_country", "device_is_new", "device_shared_flag", "ip_shared_flag",
        "device_customer_count", "ip_customer_count",
        "velocity_5min", "velocity_1hr", "seconds_since_prev_txn",
        "failed_txn_count_24h", "historical_fraud_rate",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    return df, feature_cols
