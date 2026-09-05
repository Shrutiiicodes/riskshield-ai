# RiskShield AI: Intelligent Fraud Prevention & Investigation Platform

RiskShield AI is an enterprise-grade end-to-end fraud detection and investigation platform. It combines supervised XGBoost classification, unsupervised Isolation Forest anomaly detection, graph-based NetworkX abuse-ring detection, SHAP-grounded LLM evidence reasoning agents, and dynamic policy guardrails.

---

## 🏗️ System Architecture & Directory Structure

```text
riskshield-ai/
├── README.md                              # System documentation & setup guide
├── .gitignore                             # Git ignore rules for artifacts & python
├── requirements.txt                       # Production & development dependencies
├── Dockerfile                             # Container configuration for backend services
├── docker-compose.yml                     # Multi-container service orchestration
│
├── data/
│   ├── generate_data.py                   # Synthetic transaction generator (6 fraud scenarios)
│   └── processed/                         # Generated dataset (97,424 transactions)
│       ├── transactions.csv
│       ├── transactions_scenario_labels.csv
│       ├── customers.csv
│       ├── devices.csv
│       ├── ips.csv
│       ├── merchants.csv
│       └── features_sample.csv
│
├── notebooks/
│   ├── 01_data_generation.ipynb           # Synthetic generator inspection & data sanity checks
│   ├── 02_eda.ipynb                       # Class imbalance & transaction distributions
│   ├── 03_feature_engineering.ipynb       # Velocity, deviation & novelty feature engineering
│   ├── 04_model_training.ipynb            # XGBoost & Isolation Forest model training
│   ├── 05_evaluation.ipynb                # Scored test set metrics & plot inspection
│   └── _build_notebooks.py                # Re-generates and executes notebooks
│
├── backend/
│   ├── main.py                            # FastAPI app entrypoint
│   ├── state.py                           # Application state & once-loaded ML models
│   ├── api/                               # REST API endpoints (transactions, risk, cases, analytics)
│   ├── features/                          # Feature engineering pipelines
│   ├── models/                            # Fraud classification & anomaly detection models
│   ├── graph/                             # NetworkX fraud ring detector
│   ├── agents/                            # SHAP-grounded AI investigation agent
│   └── policies/                          # Deterministic risk policy engine
│
├── dashboard/
│   └── app.py                             # Streamlit interactive operational dashboard
│
├── evaluation/
│   ├── metrics.py                         # Operational, financial & ML evaluation metrics
│   └── reports/                           # Pipeline evaluation reports & artifacts
│
├── tests/                                 # Unit & integration tests
└── run_pipeline.py                        # End-to-end automated orchestration pipeline
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+
- Docker & Docker Compose (optional for containerized deployment)

### 2. Environment Setup
```bash
# Clone repository
cd riskshield-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline & Start Services
```bash
# Run end-to-end data generation, feature engineering, model training, and evaluation
python run_pipeline.py

# Start Backend REST API
uvicorn backend.main:app --reload --port 8000

# Start Dashboard (in another terminal)
streamlit run dashboard/app.py
```

---

## 🐳 Docker Deployment

To launch all services with Docker Compose:

```bash
docker-compose up --build
```
- **FastAPI API Server**: http://localhost:8000
- **Streamlit Dashboard**: http://localhost:8501
