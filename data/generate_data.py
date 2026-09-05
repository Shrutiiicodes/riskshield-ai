"""
RiskShield AI - Synthetic Data Generator
Generates realistic merchant transaction data with embedded fraud scenarios:
  A. Velocity attacks
  B. Account takeover
  C. Card testing
  D. Device sharing / abuse rings
  E. Coordinated abuse rings (multi-account)
  F. Geographic anomalies (impossible travel)

Output (data/processed/):
  transactions.csv, customers.csv, devices.csv, ips.csv, merchants.csv
"""
import numpy as np
import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 8000
N_MERCHANTS = 40
N_LEGIT_TXNS = 92000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 10, 31)  # Jan-Aug train, Sep val, Oct test (temporal split)

CITIES = [
    ("Mumbai", "IN"), ("Delhi", "IN"), ("Bangalore", "IN"), ("Chennai", "IN"),
    ("Pune", "IN"), ("Hyderabad", "IN"), ("Kolkata", "IN"), ("Ahmedabad", "IN"),
    ("Singapore", "SG"), ("Dubai", "AE"), ("London", "GB"), ("New York", "US"),
]
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
MERCHANT_CATEGORIES = ["ecommerce", "travel", "food", "electronics", "subscription", "gaming"]


def rand_time(start, end):
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# 1. Base entities
# ---------------------------------------------------------------------------
print("Generating base entities...")

merchants = []
for i in range(N_MERCHANTS):
    cat = random.choice(MERCHANT_CATEGORIES)
    merchants.append({
        "merchant_id": f"MER_{i:04d}",
        "category": cat,
        "avg_transaction_value": round(np.random.lognormal(mean=6.5, sigma=0.6), 2),
        "daily_volume": random.randint(50, 3000),
        "historical_fraud_rate": round(np.random.uniform(0.002, 0.03), 4),
    })
merchants_df = pd.DataFrame(merchants)

customers = []
devices = []
ips = []
customer_home = {}  # customer_id -> (city, country)
customer_device = {}  # primary device
customer_ip = {}      # primary ip

for i in range(N_CUSTOMERS):
    cid = f"CUST_{i:05d}"
    home_city, home_country = random.choice(CITIES[:8])  # customers mostly based in India
    account_age = random.randint(1, 2000)  # days
    avg_amt = round(np.random.lognormal(mean=6.2, sigma=0.7), 2)
    freq = round(np.random.gamma(shape=2.0, scale=1.5), 2)  # txns/week baseline
    customers.append({
        "customer_id": cid,
        "account_age_days": account_age,
        "avg_transaction_amount": avg_amt,
        "transaction_frequency_per_week": freq,
        "historical_chargebacks": 0,
        "historical_refunds": 0,
        "home_city": home_city,
        "home_country": home_country,
    })
    did = new_id("DEV")
    ipid = new_id("IP")
    devices.append({"device_id": did, "first_seen": None, "customer_count": 1, "transaction_count": 0})
    ips.append({"ip_id": ipid, "country": home_country, "customer_count": 1, "transaction_count": 0})
    customer_home[cid] = (home_city, home_country)
    customer_device[cid] = did
    customer_ip[cid] = ipid

customers_df = pd.DataFrame(customers).set_index("customer_id")
device_index = {d["device_id"]: d for d in devices}
ip_index = {d["ip_id"]: d for d in ips}

# ---------------------------------------------------------------------------
# 2. Legitimate transaction generation
# ---------------------------------------------------------------------------
print(f"Generating {N_LEGIT_TXNS} legitimate transactions...")

txns = []


def make_txn(customer_id, merchant_id, amount, ts, device_id, ip_id, city, country,
             payment_method, status="success", label=0, scenario="legit", failure_reason=None):
    return {
        "transaction_id": new_id("TXN"),
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "timestamp": ts,
        "amount": round(max(amount, 1.0), 2),
        "currency": "INR",
        "payment_method": payment_method,
        "device_id": device_id,
        "ip_id": ip_id,
        "country": country,
        "city": city,
        "transaction_status": status,
        "failure_reason": failure_reason,
        "is_chargeback": 0,
        "is_refund": 1 if (status == "success" and random.random() < 0.03) else 0,
        "label": label,          # 1 = fraud
        "scenario": scenario,    # ground-truth scenario tag (for analysis only)
    }


customer_ids = list(customers_df.index)
merchant_ids = merchants_df["merchant_id"].tolist()

for _ in range(N_LEGIT_TXNS):
    cid = random.choice(customer_ids)
    crow = customers_df.loc[cid]
    mid = random.choice(merchant_ids)
    ts = rand_time(START_DATE, END_DATE)
    amount = max(np.random.normal(crow["avg_transaction_amount"], crow["avg_transaction_amount"] * 0.25), 20)
    city, country = customer_home[cid]
    device_id = customer_device[cid]
    ip_id = customer_ip[cid]
    pm = random.choice(PAYMENT_METHODS)
    status = "success" if random.random() > 0.03 else "failed"
    failure_reason = random.choice(["insufficient_funds", "bank_decline", "timeout"]) if status == "failed" else None
    txns.append(make_txn(cid, mid, amount, ts, device_id, ip_id, city, country, pm, status,
                          label=0, scenario="legit", failure_reason=failure_reason))

# ---------------------------------------------------------------------------
# 3. Fraud scenario injection
# ---------------------------------------------------------------------------

# --- Scenario A: Velocity attack (rapid-fire transactions from one customer) ---
print("Injecting Scenario A: velocity attacks...")
n_velocity_rings = 220
for _ in range(n_velocity_rings):
    cid = random.choice(customer_ids)
    crow = customers_df.loc[cid]
    mid = random.choice(merchant_ids)
    base_ts = rand_time(START_DATE, END_DATE)
    device_id = customer_device[cid]
    ip_id = customer_ip[cid]
    city, country = customer_home[cid]
    n = random.randint(6, 12)
    for k in range(n):
        ts = base_ts + timedelta(seconds=random.randint(5, 45) * k)
        amt = crow["avg_transaction_amount"] * random.uniform(0.8, 1.5)
        txns.append(make_txn(cid, mid, amt, ts, device_id, ip_id, city, country,
                              random.choice(PAYMENT_METHODS), "success", label=1, scenario="velocity_attack"))

# --- Scenario B: Account takeover (sudden device/geo/amount shift) ---
print("Injecting Scenario B: account takeover...")
n_ato = 260
for _ in range(n_ato):
    cid = random.choice(customer_ids)
    crow = customers_df.loc[cid]
    mid = random.choice(merchant_ids)
    ts = rand_time(START_DATE, END_DATE)
    new_device = new_id("DEV")
    device_index[new_device] = {"device_id": new_device, "first_seen": ts, "customer_count": 1, "transaction_count": 0}
    new_city, new_country = random.choice(CITIES[8:])  # foreign / unusual city
    new_ipid = new_id("IP")
    ip_index[new_ipid] = {"ip_id": new_ipid, "country": new_country, "customer_count": 1, "transaction_count": 0}
    amt = crow["avg_transaction_amount"] * random.uniform(5, 15)
    n = random.randint(1, 3)
    for k in range(n):
        txns.append(make_txn(cid, mid, amt, ts + timedelta(minutes=k * 3), new_device, new_ipid,
                              new_city, new_country, "card", "success", label=1, scenario="account_takeover"))

# --- Scenario C: Card testing (many tiny amounts, then a big one) ---
print("Injecting Scenario C: card testing...")
n_card_testing = 200
for _ in range(n_card_testing):
    cid = random.choice(customer_ids)
    mid = random.choice(merchant_ids)
    base_ts = rand_time(START_DATE, END_DATE)
    device_id = customer_device[cid]
    ip_id = customer_ip[cid]
    city, country = customer_home[cid]
    small_amounts = [round(random.uniform(5, 60), 2) for _ in range(random.randint(5, 9))]
    for k, amt in enumerate(small_amounts):
        ts = base_ts + timedelta(seconds=random.randint(10, 60) * k)
        status = "success" if random.random() > 0.4 else "failed"
        txns.append(make_txn(cid, mid, amt, ts, device_id, ip_id, city, country, "card", status,
                              label=1, scenario="card_testing",
                              failure_reason=None if status == "success" else "card_declined"))
    big_amt = round(random.uniform(2000, 8000), 2)
    txns.append(make_txn(cid, mid, big_amt, base_ts + timedelta(minutes=15), device_id, ip_id,
                          city, country, "card", "success", label=1, scenario="card_testing"))

# --- Scenario D + E: Device sharing / coordinated abuse rings ---
print("Injecting Scenario D/E: device sharing & coordinated abuse rings...")
n_rings = 55
for r in range(n_rings):
    ring_size = random.randint(5, 10)
    ring_customers = random.sample(customer_ids, ring_size)
    shared_devices = [new_id("DEV") for _ in range(random.randint(1, 3))]
    shared_ips = [new_id("IP") for _ in range(random.randint(1, 2))]
    for did in shared_devices:
        device_index[did] = {"device_id": did, "first_seen": None, "customer_count": ring_size, "transaction_count": 0}
    for ipid in shared_ips:
        ip_index[ipid] = {"ip_id": ipid, "country": "IN", "customer_count": ring_size, "transaction_count": 0}
    mid = random.choice(merchant_ids)
    ring_window_start = rand_time(START_DATE, END_DATE)
    for cid in ring_customers:
        device_id = random.choice(shared_devices)
        ip_id = random.choice(shared_ips)
        city, country = random.choice(CITIES[:8])
        n_txn = random.randint(1, 4)
        for k in range(n_txn):
            ts = ring_window_start + timedelta(hours=random.uniform(0, 72))
            amt = round(random.uniform(300, 4000), 2)
            txns.append(make_txn(cid, mid, amt, ts, device_id, ip_id, city, country,
                                  random.choice(PAYMENT_METHODS), "success", label=1, scenario="abuse_ring"))

# --- Scenario F: Geographic anomaly (impossible travel) ---
print("Injecting Scenario F: geographic anomalies...")
n_geo = 150
for _ in range(n_geo):
    cid = random.choice(customer_ids)
    crow = customers_df.loc[cid]
    mid = random.choice(merchant_ids)
    home_city, home_country = customer_home[cid]
    ts1 = rand_time(START_DATE, END_DATE - timedelta(hours=2))
    ts2 = ts1 + timedelta(minutes=random.randint(10, 90))  # impossible travel window
    far_city, far_country = random.choice([c for c in CITIES if c[1] != home_country])
    device_id = customer_device[cid]
    ip_id = customer_ip[cid]
    new_ipid = new_id("IP")
    ip_index[new_ipid] = {"ip_id": new_ipid, "country": far_country, "customer_count": 1, "transaction_count": 0}
    txns.append(make_txn(cid, mid, crow["avg_transaction_amount"], ts1, device_id, ip_id,
                          home_city, home_country, "card", "success", label=0, scenario="legit"))
    txns.append(make_txn(cid, mid, crow["avg_transaction_amount"] * random.uniform(2, 6), ts2,
                          new_id("DEV"), new_ipid, far_city, far_country, "card", "success",
                          label=1, scenario="geo_anomaly"))

# ---------------------------------------------------------------------------
# 4. Assemble & finalize
# ---------------------------------------------------------------------------
print("Assembling final dataset...")
txns_df = pd.DataFrame(txns)
txns_df = txns_df.sort_values("timestamp").reset_index(drop=True)

# chargebacks: some fraud txns get flagged as chargebacks after the fact
fraud_idx = txns_df[txns_df.label == 1].sample(frac=0.35, random_state=RNG_SEED).index
txns_df.loc[fraud_idx, "is_chargeback"] = 1

# update customer historical chargeback/refund counts
cb_counts = txns_df[txns_df.is_chargeback == 1].groupby("customer_id").size()
rf_counts = txns_df[txns_df.is_refund == 1].groupby("customer_id").size()
customers_df["historical_chargebacks"] = customers_df.index.map(cb_counts).fillna(0).astype(int)
customers_df["historical_refunds"] = customers_df.index.map(rf_counts).fillna(0).astype(int)

# finalize device / ip tables from actual usage
dev_usage = txns_df.groupby("device_id").agg(
    transaction_count=("transaction_id", "count"),
    customer_count=("customer_id", "nunique"),
    first_seen=("timestamp", "min"),
).reset_index()
ip_usage = txns_df.groupby("ip_id").agg(
    transaction_count=("transaction_id", "count"),
    customer_count=("customer_id", "nunique"),
).reset_index()
ip_country = txns_df.groupby("ip_id")["country"].agg(lambda s: s.mode().iloc[0]).reset_index()
ip_usage = ip_usage.merge(ip_country, on="ip_id")

devices_df = dev_usage.rename(columns={"device_id": "device_id"})
ips_df = ip_usage[["ip_id", "country", "customer_count", "transaction_count"]]

customers_df = customers_df.reset_index()

# Save
txns_df.drop(columns=["scenario"]).to_csv(OUT_DIR / "transactions.csv", index=False)
txns_df[["transaction_id", "scenario"]].to_csv(OUT_DIR / "transactions_scenario_labels.csv", index=False)
customers_df.to_csv(OUT_DIR / "customers.csv", index=False)
devices_df.to_csv(OUT_DIR / "devices.csv", index=False)
ips_df.to_csv(OUT_DIR / "ips.csv", index=False)
merchants_df.to_csv(OUT_DIR / "merchants.csv", index=False)

print(f"Done. {len(txns_df):,} transactions generated "
      f"({txns_df.label.sum():,} fraud / {(txns_df.label.mean()*100):.2f}% fraud rate)")
print(txns_df.groupby("scenario").size())
