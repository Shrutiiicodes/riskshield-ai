import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.graph.ring_detector import detect_rings


def make_txn(txn_id, cust, device, ip, label):
    return {"transaction_id": txn_id, "customer_id": cust, "device_id": device,
            "ip_id": ip, "label": label}


def test_isolated_customers_form_no_cluster():
    rows = [make_txn(f"t{i}", f"cust{i}", f"dev{i}", f"ip{i}", 0) for i in range(5)]
    df = pd.DataFrame(rows)
    scored, clusters = detect_rings(df, min_customers=3)
    assert clusters.empty


def test_shared_device_forms_cluster():
    rows = []
    for i in range(5):
        rows.append(make_txn(f"t{i}", f"cust{i}", "shared_dev", f"ip{i}", 1))
    df = pd.DataFrame(rows)
    scored, clusters = detect_rings(df, min_customers=3)
    assert len(clusters) == 1
    assert clusters.iloc[0]["cluster_size"] == 5
    assert clusters.iloc[0]["cluster_fraud_rate"] == 1.0


def test_personal_device_does_not_pollute_cluster():
    # 5 customers share a fraud device (fraud), each also has their own private
    # device used for many legit purchases -- private device must not dilute purity
    rows = []
    for i in range(5):
        rows.append(make_txn(f"ring{i}", f"cust{i}", "shared_dev", "shared_ip", 1))
        for j in range(20):
            rows.append(make_txn(f"legit{i}_{j}", f"cust{i}", f"private_dev{i}", f"private_ip{i}", 0))
    df = pd.DataFrame(rows)
    scored, clusters = detect_rings(df, min_customers=3)
    assert len(clusters) == 1
    assert clusters.iloc[0]["cluster_fraud_rate"] == 1.0
