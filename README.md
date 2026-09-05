# 🛡️ RiskShield AI

**An agentic merchant-risk platform that detects transaction anomalies and coordinated fraud patterns, assigns explainable risk scores, and recommends bounded actions while minimizing costly false positives.**

Built for a fintech AI/ML risk-engineering portfolio project (Razorpay-style evaluation: held-out test set, precision/recall, false-positive cost).

> **All numbers in this README are actual measured results from running this repository's pipeline end-to-end — nothing is invented.** Re-run `python run_pipeline.py` yourself to reproduce them (fixed random seed, so results are deterministic).

---

## 1. What this is

Most "fraud detection" portfolio projects are a single classifier: `transaction -> fraud probability`. This one is closer to how a real risk team works:

```
detect → score → explain → investigate → recommend → measure
```

Two detection layers, fused together:

- **Transaction risk** — is *this* transaction suspicious? (XGBoost classifier + Isolation Forest anomaly detector on behavioral/velocity/deviation features)
- **Network risk** — are *multiple accounts* behaving like a coordinated abuse ring? (graph analysis over shared devices/IPs with NetworkX)

On top of the ML, an **investigation agent** reasons over the evidence (SHAP contributions + network context) to produce a human-readable case summary — but a separate, deterministic **policy engine** decides the actual action. The agent can recommend; it can never move money, block an account, or take an irreversible action. See [§7](#7-safety-architecture-agent-recommends-policy-decides).

---

## 2. Results (held-out test set, October data — never touched during training or threshold tuning)

| Metric | Value |
|---|---|
| Precision | **94.6%** |
| Recall | **96.5%** |
| F1 | 95.5% |
| ROC-AUC | 0.997 |
| PR-AUC | 0.982 |
| False positive rate | 0.39% |
| False positives / false negatives | 35 / 22 (out of 9,713 test transactions, 632 fraud) |

**Cost-sensitive evaluation** (the metric that actually matters — see [§6](#6-why-not-just-accuracy)):

| | Expected cost |
|---|---|
| Fused model (ML + anomaly + network), cost-optimal threshold 0.29 | **₹47,500** |
| ML-only baseline (same XGBoost model, no anomaly/network fusion) | ₹51,500 |
| **Cost reduction from fusion** | **7.8%** |

**Abuse-ring detection** (graph analysis, evaluated against ground-truth ring-fraud labels):

| Metric | Value |
|---|---|
| Suspicious clusters found (≥3 linked accounts) | 42 |
| Cluster precision | **100%** |
| Cluster recall | **99.3%** |
| Avg. cluster size | 9.8 accounts |

Full metrics, PR/ROC curves, cost-vs-threshold sweep, and feature importances are in [`evaluation/reports/`](evaluation/reports/) after running the pipeline.

---

## 3. Architecture

```
Synthetic Transactions (97,424 txns, 6 fraud scenarios)
                 │
                 ▼
        Feature Engineering
   (velocity, amount deviation, device/IP
    novelty, geo deviation, failure history)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
Transaction Risk        Network Risk
  XGBoost +          Graph analysis (NetworkX)
Isolation Forest      shared device/IP clusters
      │                     │
      └──────────┬──────────┘
                 ▼
          Risk Fusion Engine
   (weights tuned; threshold cost-optimized
          on VALIDATION set only)
                 ▼
          Risk Decision (LOW/MED/HIGH/CRITICAL)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
SHAP Explanation      Abuse Ring Context
      └──────────┬──────────┘
                 ▼
       Investigation Agent
  (reasons over evidence, produces narrative
   + a recommended action — cannot act itself)
                 ▼
        Policy / Guardrail Engine
 (deterministic; rejects any irreversible/
  monetary action the agent might request)
                 ▼
          Merchant Dashboard
   (alerts, investigation, network graph,
      model performance, audit trail)
```

**Key design principle:** ML determines risk. The LLM/agent reasons over evidence and explains it in natural language. The policy engine — not the agent — decides what actually happens. This is deliberately *not* `transaction → LLM → verdict`; that pattern is weak, non-deterministic, and hard to audit for a financial system.

---

## 4. Repository layout

```
riskshield-ai/
├── data/
│   ├── generate_data.py          # synthetic data + 6 fraud scenario generators
│   └── processed/                # generated CSVs (customers, devices, ips, merchants, transactions)
├── notebooks/
│   ├── 01_data_generation.ipynb      # runs the generator, inspects output
│   ├── 02_eda.ipynb                  # class imbalance, amount/time distributions
│   ├── 03_feature_engineering.ipynb  # builds + inspects engineered features
│   ├── 04_model_training.ipynb       # interactive XGBoost + Isolation Forest training
│   ├── 05_evaluation.ipynb           # loads pipeline results, walks through metrics
│   └── _build_notebooks.py           # (re)generates + executes all of the above
├── backend/
│   ├── main.py                   # FastAPI app entrypoint (mounts routers below)
│   ├── state.py                  # shared, once-loaded artifacts (models, reports)
│   ├── api/
│   │   ├── transactions.py       # live scoring + transaction lookup
│   │   ├── risk.py               # alerts listing + investigation
│   │   ├── cases.py              # case creation/resolution (create_case tool)
│   │   └── analytics.py          # dashboard summary/metrics/clusters
│   ├── features/engineering.py   # feature engineering
│   ├── models/fraud_models.py    # XGBoost classifier, Isolation Forest, risk fusion
│   ├── graph/ring_detector.py    # NetworkX-based abuse ring detection
│   ├── agents/investigation_agent.py  # template + optional LLM-backed investigation
│   └── policies/risk_policy.py   # deterministic guardrail / action allowlist
├── dashboard/
│   └── app.py                    # Streamlit merchant dashboard (6 pages)
├── evaluation/
│   ├── metrics.py                # classification / operational / cost / ring metrics
│   └── reports/                  # generated after running the pipeline
├── tests/
│   ├── test_policies.py
│   ├── test_graph.py
│   └── test_api.py               # FastAPI route integration tests
├── run_pipeline.py                # orchestrates the entire end-to-end run
├── requirements.txt
├── Dockerfile / docker-compose.yml
└── README.md
```

---

## 5. Running it

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (deterministic, seed=42)
python data/generate_data.py

# 2. Run the full pipeline: train models, detect rings, fuse risk,
#    tune threshold on validation set, evaluate on held-out test set,
#    generate SHAP explanations + sample agent investigations, save reports
python run_pipeline.py

# 3a. Explore results in the dashboard
streamlit run dashboard/app.py

# 3b. ...or serve them via the API
uvicorn backend.main:app --reload --port 8000
# then: curl http://localhost:8000/analytics/summary
# interactive docs at http://localhost:8000/docs

# 3c. ...or explore the build interactively via notebooks
jupyter notebook notebooks/

# Run tests (includes FastAPI route integration tests)
pytest tests/ -v
```

Or with Docker:
```bash
docker compose run pipeline     # generates data + runs the pipeline once
docker compose up api dashboard # then serves the API (8000) and dashboard (8501)
```

### Using a real LLM for investigation (optional)
By default the investigation agent uses deterministic, template-based reasoning over the same structured evidence (SHAP contributions, cluster info, related-transaction counts) — fully offline, no API key needed. Set `ANTHROPIC_API_KEY` in your environment to switch `backend/agents/investigation_agent.py` to Claude-generated narratives; it falls back to the template automatically if the call fails (see [§8](#8-failure-handling), Case 2).

---

## 6. Why not just accuracy?

The test set is 6.5% fraud. A model that predicts "not fraud" for everything scores 93.5% accuracy and catches zero fraud. Instead, this project evaluates:

- **Precision/Recall/PR-AUC** (PR-AUC matters more than ROC-AUC under class imbalance)
- **Expected cost** = `FP × cost_per_FP + FN × cost_per_FN`, using ₹100 per false positive (friction + lost legitimate revenue) and ₹2,000 per false negative (average fraud loss). These are configurable in `evaluation/metrics.py`.
- The classification **threshold is chosen by sweeping the validation set to minimize expected cost** — not fixed at 0.5 — because with asymmetric costs, the "optimal" cutoff is rarely 0.5. In this run it landed at **0.29**.

The fused model (ML + anomaly + network signals) beats an ML-only baseline on **expected cost**, not just on raw classification metrics — a materially different (and more defensible) claim.

---

## 7. Safety architecture: agent recommends, policy decides

The investigation agent's tool surface (`backend/agents/investigation_agent.py`) is intentionally missing anything that moves money or takes irreversible action:

```
Available:  get_transaction, get_customer_history, get_device_history,
            get_ip_history, get_related_transactions, get_model_explanation,
            get_cluster_information, create_case, recommend_action

Absent:     block_customer(), transfer_money(), refund_money()
```

Even if an LLM-backed agent were prompted to request one of those absent actions, `backend/policies/risk_policy.py` enforces a fixed allowlist (`none`, `enhanced_monitoring`, `create_review_case`, `recommend_hold`) and silently overrides anything outside it with the deterministic risk-tier default. This is tested in `tests/test_policies.py`.

---

## 8. Failure handling

Designed-for failure modes (see `run_pipeline.py` / `investigation_agent.py` for where these are handled):

- **Model uncertain** (risk ≈ 0.5) → routed to manual review rather than auto-approved.
- **LLM unavailable** → `investigation_agent.investigate()` catches any exception and falls back to the deterministic template narrative; ML scoring and SHAP explanations are unaffected either way.
- **Missing device/IP data** → features default to conservative values (e.g. `device_is_new=0` is *not* assumed; missing history reduces confidence rather than assuming safety).
- **Agent requests an unsafe action** → rejected by the policy engine (§7), which logs why and substitutes the safe default.

---

## 9. Synthetic data & fraud scenarios

`data/generate_data.py` generates 97,424 transactions across 8,000 customers and 40 merchants (Jan–Oct 2025, used for the temporal train/val/test split), with six explicitly modeled fraud patterns rather than randomly-flipped labels:

| Scenario | Description | Count |
|---|---|---|
| Velocity attack | 6–12 rapid transactions from one customer in minutes | 1,948 txns |
| Account takeover | Sudden new device + new country + amount 5–15× baseline | 515 txns |
| Card testing | 5–9 small probing amounts, then one large charge | 1,581 txns |
| Coordinated abuse ring | 5–10 accounts sharing 1–3 devices / 1–2 IPs | 1,080 txns |
| Geographic anomaly | Two transactions in an impossible travel window | 150 txns |
| Legitimate | Normal behavior sampled around each customer's baseline | 92,150 txns |

Overall fraud rate: 5.4%.

---

## 10. Evaluation methodology

- **Temporal split** (not random shuffling — this simulates real deployment): Jan–Aug 2025 train (78,057 txns), Sep 2025 validation (9,654 txns), Oct 2025 held-out test (9,713 txns).
- Threshold and fusion weights are chosen on the **validation set only**; the test set is touched exactly once, for final reporting.
- Graph-based ring detection links customers only through **genuinely shared** devices/IPs (used by >1 distinct customer) — a customer's own private device is deliberately excluded from the graph so that one ring transaction doesn't pull a customer's entire unrelated legitimate history into a "suspicious cluster" (this subtlety is covered by `tests/test_graph.py::test_personal_device_does_not_pollute_cluster`).

---

## 11. What I'd say in an interview

- **Why XGBoost over a deep model?** Tabular data with a moderate feature count and strong nonlinear interactions — gradient-boosted trees are the standard, well-calibrated choice, and `scale_pos_weight` handles the class imbalance without synthetic oversampling.
- **Why Isolation Forest in addition to a supervised model?** It catches anomalies that don't resemble any *labeled* historical fraud — useful for novel attack patterns a supervised model has never seen.
- **Why is accuracy insufficient here?** 6.5% fraud rate means a trivial always-negative model scores 93.5% "accuracy" while catching nothing; PR-AUC and cost-weighted metrics are the honest yardstick.
- **How is data leakage avoided?** Strict temporal split; the classifier and IsolationForest are fit only on Jan–Aug data, the threshold is tuned only on September, and October is scored exactly once.
- **How are fraud rings detected without a heavy GNN?** A NetworkX connected-components pass over a bipartite customer↔shared-device/IP graph is enough to get 100% cluster precision / 99.3% recall here — added complexity (e.g. a GNN) would be justified only if a real deployment showed this approach saturating.
- **Why keep the LLM agent out of the classification decision?** Determinism, auditability, and cost — an LLM call per transaction doesn't scale to payment volumes and isn't the right tool for structured tabular classification anyway. Its value is turning risk scores + SHAP values + network context into a readable investigation narrative for a human analyst.
