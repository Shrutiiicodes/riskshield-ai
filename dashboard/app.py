"""
RiskShield AI - Merchant Dashboard (Streamlit)

Run:
    streamlit run dashboard/app.py

Reads pre-computed results from evaluation/reports/ (produced by
run_pipeline.py at the repo root) -- no live backend required, though it
will also work against the FastAPI backend if RISKSHIELD_API_URL is set.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
REPORT_DIR = ROOT / "evaluation" / "reports"
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="RiskShield AI", layout="wide", page_icon="🛡️")


@st.cache_data
def load_data():
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())
    test_df = pd.read_csv(REPORT_DIR / "test_set_scored.csv")
    clusters_df = pd.read_csv(REPORT_DIR / "clusters.csv")
    return metrics, test_df, clusters_df


if not (REPORT_DIR / "metrics.json").exists():
    st.error("No results found. Run `python run_pipeline.py` from the project root first.")
    st.stop()

metrics, test_df, clusters_df = load_data()

st.title("🛡️ RiskShield AI")
st.caption("Agentic merchant-risk platform — transaction anomaly detection, coordinated abuse-ring "
           "discovery, explainable risk scoring, and cost-aware evaluation on a held-out test set.")

page = st.sidebar.radio("Navigate", [
    "Overview", "Live Alerts", "Investigation", "Abuse Network", "Model Performance", "Audit Trail"
])

# ---------------------------------------------------------------------------
# PAGE 1: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    c = metrics["classification_metrics"]
    op = metrics["operational_metrics"]
    cost = metrics["cost_analysis"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions analyzed (test set)", f"{metrics['temporal_split']['test_txns']:,}")
    col2.metric("Fraud detected", f"{op['true_positives']:,}")
    col3.metric("False positives", f"{op['false_positives']:,}")
    col4.metric("Suspicious clusters found", metrics["ring_detection_metrics"]["clusters_found"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{c['precision']*100:.1f}%")
    col2.metric("Recall", f"{c['recall']*100:.1f}%")
    col3.metric("PR-AUC", f"{c['pr_auc']:.3f}")
    col4.metric("ROC-AUC", f"{c['roc_auc']:.3f}")

    st.markdown("### 💰 Financial impact (held-out test set)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected fraud loss (missed fraud)", f"₹{cost['expected_fn_cost_inr']:,}")
    col2.metric("Expected false-positive cost", f"₹{cost['expected_fp_cost_inr']:,}")
    col3.metric("Total expected cost", f"₹{cost['total_expected_cost_inr']:,}")

    st.info(
        f"Cost-optimal threshold (tuned on validation set only): **{metrics['optimal_threshold']}**. "
        f"The fused model reduces total expected cost by **{metrics['cost_reduction_vs_ml_only_baseline_pct']}%** "
        f"vs. an ML-only baseline (same XGBoost model, no anomaly/network fusion) evaluated on the same test set."
    )

    st.markdown("### Risk score distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=test_df[test_df.label == 0]["risk_score"], name="Legitimate",
                                opacity=0.6, histnorm="probability density", nbinsx=40))
    fig.add_trace(go.Histogram(x=test_df[test_df.label == 1]["risk_score"], name="Fraud",
                                opacity=0.6, histnorm="probability density", nbinsx=40))
    fig.update_layout(barmode="overlay", xaxis_title="Fused risk score", yaxis_title="Density", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2: Live Alerts
# ---------------------------------------------------------------------------
elif page == "Live Alerts":
    st.markdown("### Flagged transactions")
    risk_filter = st.multiselect("Risk level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                                  default=["CRITICAL", "HIGH"])
    filtered = test_df[test_df["risk_level"].isin(risk_filter)].sort_values("risk_score", ascending=False)
    st.write(f"{len(filtered):,} transactions match")

    icon_map = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    for _, row in filtered.head(25).iterrows():
        icon = icon_map.get(row["risk_level"], "⚪")
        with st.expander(f"{icon} {row['risk_level']}  |  {row['transaction_id']}  |  ₹{row['amount']:,.0f}  "
                          f"|  risk={row['risk_score']:.2f}"):
            cols = st.columns(3)
            cols[0].write(f"**Customer:** {row['customer_id']}")
            cols[1].write(f"**Merchant:** {row['merchant_id']}")
            cols[2].write(f"**Ground truth:** {'FRAUD' if row['label'] else 'legit'} "
                          f"({row.get('scenario', 'n/a')})")
            if st.button("Investigate", key=f"inv_{row['transaction_id']}"):
                st.session_state["selected_txn"] = row["transaction_id"]
                st.info("Open the 'Investigation' tab to see the full case.")

# ---------------------------------------------------------------------------
# PAGE 3: Investigation
# ---------------------------------------------------------------------------
elif page == "Investigation":
    from backend.features.engineering import load_raw
    from backend.agents.investigation_agent import investigate, build_evidence_reasons
    from backend.policies.risk_policy import evaluate_policy
    import joblib
    import shap

    txn_ids = test_df.sort_values("risk_score", ascending=False)["transaction_id"].head(200).tolist()
    default_idx = 0
    if "selected_txn" in st.session_state and st.session_state["selected_txn"] in txn_ids:
        default_idx = txn_ids.index(st.session_state["selected_txn"])
    txn_id = st.selectbox("Transaction", txn_ids, index=default_idx)

    row = test_df[test_df["transaction_id"] == txn_id].iloc[0]
    feature_cols = json.loads((REPORT_DIR / "feature_cols.json").read_text())

    st.markdown(f"## Transaction `{txn_id}`")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{row['risk_score']*100:.0f}%")
    col2.metric("Amount", f"₹{row['amount']:,.0f}")
    col3.metric("Risk Level", row["risk_level"])

    clf = joblib.load(REPORT_DIR / "fraud_classifier.joblib")
    explainer = shap.TreeExplainer(clf.model)
    shap_vals = explainer.shap_values(pd.DataFrame([row[feature_cols]]))[0]
    contrib = dict(zip(feature_cols, shap_vals))
    reasons = build_evidence_reasons(row, contrib)

    st.markdown("### Why was it flagged?")
    for r in reasons:
        val = float(min(1.0, abs(r["contribution"]) / (max(abs(v) for v in contrib.values()) + 1e-9)))
        st.progress(val, text=f"**{r['description']}**  (SHAP contribution: {r['contribution']:+.3f})")

    cluster_info = None
    if row.get("cluster_id", 0) and row["cluster_id"] > 0:
        crow = clusters_df[clusters_df["cluster_id"] == row["cluster_id"]]
        if not crow.empty:
            cluster_info = {"cluster_size": int(crow.iloc[0]["cluster_size"]),
                             "shared_device_count": int(crow.iloc[0]["shared_device_count"])}
            st.markdown("### Network")
            st.write(f"This customer's device/IP is linked to **{cluster_info['cluster_size']} accounts** "
                      f"through **{cluster_info['shared_device_count']} shared device(s)**.")

    evidence = {
        "transaction": {"transaction_id": row["transaction_id"], "amount": float(row["amount"]),
                         "customer_id": row["customer_id"], "merchant_id": row["merchant_id"]},
        "risk_score": float(row["risk_score"]), "risk_level": row["risk_level"],
        "reasons": reasons, "cluster": cluster_info,
        "related_transaction_count": int((test_df["customer_id"] == row["customer_id"]).sum()),
    }
    investigation = investigate(evidence)
    policy_decision = evaluate_policy(row["risk_level"], investigation["recommended_action"])

    st.markdown("### 🤖 AI Investigation")
    st.write(investigation["summary"])
    st.caption(f"Agent confidence: {investigation['confidence']*100:.0f}%  |  reasoning mode: {investigation['mode']}")

    st.markdown("### Recommendation")
    st.warning(f"**Agent recommends:** {investigation['recommended_action'].replace('_', ' ').title()}")
    st.success(f"**Policy engine decision:** {policy_decision.action.replace('_', ' ').title()}  \n"
               f"_{policy_decision.reason}_")

    with st.expander("Ground truth (for demo/evaluation purposes only)"):
        st.write(f"Label: {'FRAUD' if row['label'] else 'legitimate'}  |  Scenario: {row.get('scenario')}")

# ---------------------------------------------------------------------------
# PAGE 4: Abuse Network
# ---------------------------------------------------------------------------
elif page == "Abuse Network":
    st.markdown("### Coordinated abuse clusters")
    st.write(f"**{len(clusters_df)}** clusters found where 3+ customer accounts share a device or IP.")
    st.dataframe(
        clusters_df[["cluster_id", "cluster_size", "shared_device_count", "shared_ip_count", "cluster_fraud_rate"]]
        .sort_values("cluster_size", ascending=False),
        use_container_width=True,
    )

    if not clusters_df.empty:
        top_cluster_id = clusters_df.sort_values("cluster_size", ascending=False).iloc[0]["cluster_id"]
        st.markdown(f"### Cluster #{int(top_cluster_id)} network graph")

        import networkx as nx
        from backend.features.engineering import load_raw
        from backend.graph.ring_detector import build_entity_graph

        txns, customers, devices, ips, merchants = load_raw(REPORT_DIR.parent.parent / "data" / "processed")
        G = build_entity_graph(txns)
        components = list(nx.connected_components(G))
        target_customers = set(eval(clusters_df[clusters_df.cluster_id == top_cluster_id].iloc[0]["customers"]))
        comp = next((c for c in components if any(n[0] == "customer" and n[1] in target_customers for n in c)), None)

        if comp:
            subG = G.subgraph(comp)
            pos = nx.spring_layout(subG, seed=42, k=0.6)
            edge_x, edge_y = [], []
            for e in subG.edges():
                edge_x += [pos[e[0]][0], pos[e[1]][0], None]
                edge_y += [pos[e[0]][1], pos[e[1]][1], None]
            node_x, node_y, node_color, node_text = [], [], [], []
            for n in subG.nodes():
                node_x.append(pos[n][0]); node_y.append(pos[n][1])
                kind = n[0]
                node_color.append({"customer": "#4C78A8", "device": "#F58518", "ip": "#E45756"}.get(kind, "gray"))
                node_text.append(f"{kind}: {n[1]}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#ccc"), hoverinfo="none"))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(size=14, color=node_color),
                                      text=node_text, hoverinfo="text"))
            fig.update_layout(showlegend=False, height=550,
                               xaxis=dict(visible=False), yaxis=dict(visible=False),
                               title="Blue = customer, Orange = device, Red = IP")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Multiple customer accounts converging on the same device/IP nodes is the signature "
                       "of a coordinated abuse ring rather than independent fraud.")

# ---------------------------------------------------------------------------
# PAGE 5: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown("### Held-out test set performance")
    c, op, cost = metrics["classification_metrics"], metrics["operational_metrics"], metrics["cost_analysis"]
    perf_table = pd.DataFrame([
        {"Metric": "Precision", "Value": f"{c['precision']*100:.1f}%"},
        {"Metric": "Recall", "Value": f"{c['recall']*100:.1f}%"},
        {"Metric": "F1", "Value": f"{c['f1']*100:.1f}%"},
        {"Metric": "ROC-AUC", "Value": f"{c['roc_auc']:.3f}"},
        {"Metric": "PR-AUC", "Value": f"{c['pr_auc']:.3f}"},
        {"Metric": "False Positive Rate", "Value": f"{op['false_positive_rate']*100:.2f}%"},
        {"Metric": "False Negatives", "Value": op["false_negatives"]},
        {"Metric": "False Positives", "Value": op["false_positives"]},
        {"Metric": "Expected FP Cost", "Value": f"₹{cost['expected_fp_cost_inr']:,}"},
        {"Metric": "Expected FN Cost", "Value": f"₹{cost['expected_fn_cost_inr']:,}"},
        {"Metric": "Total Expected Cost", "Value": f"₹{cost['total_expected_cost_inr']:,}"},
    ])
    st.table(perf_table)

    st.markdown("### Evaluation plots (PR curve, ROC curve, cost-vs-threshold, score distribution)")
    st.image(str(REPORT_DIR / "evaluation_plots.png"))

    st.markdown("### Feature importance")
    st.image(str(REPORT_DIR / "feature_importance.png"))

# ---------------------------------------------------------------------------
# PAGE 6: Audit Trail
# ---------------------------------------------------------------------------
elif page == "Audit Trail":
    st.markdown("### Sample decision audit trail")
    st.caption("Every scored transaction produces a traceable chain: risk scoring → SHAP explanation → "
               "network check → AI investigation → policy decision. Below are the top flagged cases from "
               "this run.")
    for ex in metrics.get("sample_investigations", []):
        with st.expander(f"{ex['risk_level']}  |  {ex['transaction_id']}  |  ₹{ex['amount']:,.0f}  "
                          f"|  score={ex['risk_score']:.2f}  |  truth={'FRAUD' if ex['ground_truth_label'] else 'legit'}"):
            st.write("**Evidence:**")
            for r in ex["evidence_reasons"]:
                st.write(f"- {r['description']}  (contribution: {r['contribution']:+.3f})")
            st.write(f"**AI investigation summary:** {ex['investigation_summary']}")
            st.write(f"**Agent recommended action:** `{ex['agent_recommended_action']}` "
                     f"(confidence {ex['agent_confidence']*100:.0f}%)")
            st.write(f"**Policy engine final action:** `{ex['policy_final_action']}`")
            st.caption(ex["policy_reason"])
