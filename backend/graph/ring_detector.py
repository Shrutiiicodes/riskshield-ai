"""
RiskShield AI - Abuse Ring Detector
Builds a bipartite-ish graph of customers <-> devices <-> ips and finds
connected components / clusters where multiple customers share the same
device or IP -- a strong signal of coordinated abuse (device farms, mule
accounts, promo abuse rings, etc).
"""
import networkx as nx
import pandas as pd


def build_entity_graph(txns):
    """
    Build an undirected graph linking customers via GENUINELY SHARED device/ip
    nodes only (used by more than one distinct customer). A customer's own
    personal device/IP used solely by them carries no network signal and is
    deliberately excluded -- otherwise every customer's private device would
    bridge their legitimate activity into any ring they're incidentally part
    of, diluting cluster purity.
    """
    dev_cust_counts = txns.groupby("device_id")["customer_id"].nunique()
    ip_cust_counts = txns.groupby("ip_id")["customer_id"].nunique()
    shared_devices = set(dev_cust_counts[dev_cust_counts > 1].index)
    shared_ips = set(ip_cust_counts[ip_cust_counts > 1].index)

    G = nx.Graph()
    shared_rows = txns[txns["device_id"].isin(shared_devices) | txns["ip_id"].isin(shared_ips)]
    for _, row in shared_rows.iterrows():
        cust_node = ("customer", row["customer_id"])
        G.add_node(cust_node, kind="customer")
        if row["device_id"] in shared_devices:
            dev_node = ("device", row["device_id"])
            G.add_node(dev_node, kind="device")
            G.add_edge(cust_node, dev_node)
        if row["ip_id"] in shared_ips:
            ip_node = ("ip", row["ip_id"])
            G.add_node(ip_node, kind="ip")
            G.add_edge(cust_node, ip_node)
    return G


def detect_rings(txns, min_customers=3):
    """
    Find connected components with >= min_customers distinct customer nodes.
    Returns a DataFrame: cluster_id, customer_id, cluster_size, shared_devices, shared_ips
    plus a per-transaction network_risk_score.
    """
    G = build_entity_graph(txns)
    components = list(nx.connected_components(G))

    cluster_rows = []
    customer_to_cluster = {}
    cluster_id = 0
    for comp in components:
        customers_in_comp = [n[1] for n in comp if n[0] == "customer"]
        devices_in_comp = [n[1] for n in comp if n[0] == "device"]
        ips_in_comp = [n[1] for n in comp if n[0] == "ip"]
        if len(customers_in_comp) >= min_customers:
            cluster_id += 1
            for cid in customers_in_comp:
                customer_to_cluster[cid] = cluster_id
            cluster_rows.append({
                "cluster_id": cluster_id,
                "cluster_size": len(customers_in_comp),
                "customers": customers_in_comp,
                "devices": devices_in_comp,
                "ips": ips_in_comp,
                "shared_device_count": len(devices_in_comp),
                "shared_ip_count": len(ips_in_comp),
            })

    clusters_df = pd.DataFrame(cluster_rows)

    # Cluster "purity": fraud rate measured on the transactions that actually flow
    # through this cluster's shared devices/ips (the ring activity itself), not a
    # customer's entire lifetime history -- a customer can be legitimate 95% of the
    # time and still take part in one abusive ring transaction.
    if not clusters_df.empty and "label" in txns.columns:
        def ring_fraud_rate(row):
            dev_set, ip_set = set(row["devices"]), set(row["ips"])
            mask = txns["device_id"].isin(dev_set) | txns["ip_id"].isin(ip_set)
            ring_txns = txns[mask]
            return round(float(ring_txns["label"].mean()), 4) if len(ring_txns) else 0.0

        clusters_df["cluster_fraud_rate"] = clusters_df.apply(ring_fraud_rate, axis=1)
        clusters_df = clusters_df.drop(columns=["devices", "ips"])

    # per-transaction network risk score: based on the cluster size / shared entity fan-out
    txns = txns.copy()
    txns["cluster_id"] = txns["customer_id"].map(customer_to_cluster).fillna(0).astype(int)
    cluster_size_map = clusters_df.set_index("cluster_id")["cluster_size"].to_dict() if not clusters_df.empty else {}
    txns["cluster_size"] = txns["cluster_id"].map(cluster_size_map).fillna(1).astype(int)

    def network_score(row):
        if row["cluster_id"] == 0:
            base = 0.0
        else:
            # scale with log of cluster size, cap at 1.0
            import math
            base = min(1.0, math.log2(max(row["cluster_size"], 2)) / 5.0)
        return round(base, 4)

    txns["network_risk_score"] = txns.apply(network_score, axis=1)

    return txns, clusters_df
