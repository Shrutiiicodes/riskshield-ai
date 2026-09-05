"""
Generates the five project notebooks as real, executable notebooks (not
stubs), then runs them so committed outputs/plots are genuine. Run once
from the repo root: `python notebooks/_build_notebooks.py`
"""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

NB_DIR = Path(__file__).parent
ROOT = NB_DIR.parent


def make_notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build_and_run(name, cells):
    nb = make_notebook(cells)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    path = NB_DIR / name
    nbf.write(nb, path)
    print(f"Built + executed: {path}")


# ---------------------------------------------------------------------------
# 01: Data generation
# ---------------------------------------------------------------------------
build_and_run("01_data_generation.ipynb", [
    md("# 01 - Synthetic Data Generation\n\n"
       "Generates the RiskShield AI transaction dataset used throughout the rest of the pipeline. "
       "See `data/generate_data.py` for the full generator; this notebook runs it and inspects the "
       "output. Six fraud scenarios are embedded deliberately (not random label flips) so the ML "
       "models have learnable structure to find."),
    code("import subprocess, sys\n"
         "sys.path.insert(0, '.')\n"
         "result = subprocess.run([sys.executable, 'data/generate_data.py'], capture_output=True, text=True)\n"
         "print(result.stdout)\n"
         "print(result.stderr)"),
    md("## Inspect the generated files"),
    code("import pandas as pd\n"
         "txns = pd.read_csv('data/processed/transactions.csv')\n"
         "scenario_labels = pd.read_csv('data/processed/transactions_scenario_labels.csv')\n"
         "txns = txns.merge(scenario_labels, on='transaction_id')\n"
         "print(f'{len(txns):,} transactions')\n"
         "print(f'Fraud rate: {txns.label.mean()*100:.2f}%')\n"
         "txns.groupby('scenario').size().sort_values(ascending=False)"),
    code("customers = pd.read_csv('data/processed/customers.csv')\n"
         "devices = pd.read_csv('data/processed/devices.csv')\n"
         "ips = pd.read_csv('data/processed/ips.csv')\n"
         "merchants = pd.read_csv('data/processed/merchants.csv')\n"
         "print(f'Customers: {len(customers):,}  Devices: {len(devices):,}  IPs: {len(ips):,}  Merchants: {len(merchants):,}')"),
    md("## Sanity check: fraud scenarios look distinguishable from legit traffic"),
    code("txns.groupby('scenario')['amount'].describe()[['mean', 'std', 'min', 'max']]"),
])

# ---------------------------------------------------------------------------
# 02: EDA
# ---------------------------------------------------------------------------
build_and_run("02_eda.ipynb", [
    md("# 02 - Exploratory Data Analysis\n\n"
       "Basic exploration of the generated transaction data before feature engineering: "
       "class imbalance, amount distributions, time-of-day/velocity patterns, and how fraud "
       "scenarios differ from legitimate traffic."),
    code("import pandas as pd\n"
         "import matplotlib.pyplot as plt\n"
         "import matplotlib\n"
         "matplotlib.use('Agg')\n"
         "%matplotlib inline\n\n"
         "txns = pd.read_csv('data/processed/transactions.csv')\n"
         "txns['timestamp'] = pd.to_datetime(txns['timestamp'], format='mixed')\n"
         "scenario_labels = pd.read_csv('data/processed/transactions_scenario_labels.csv')\n"
         "txns = txns.merge(scenario_labels, on='transaction_id')\n"
         "txns.shape"),
    md("## Class imbalance\n\nThis is why accuracy is a bad metric here -- see `evaluation/metrics.py`."),
    code("fraud_rate = txns.label.mean()\n"
         "print(f'Fraud rate: {fraud_rate*100:.2f}%  ({txns.label.sum():,} of {len(txns):,} transactions)')\n"
         "print(f'A model predicting \"never fraud\" would score {(1-fraud_rate)*100:.1f}% accuracy while catching zero fraud.')"),
    md("## Amount distribution: fraud vs legit"),
    code("fig, ax = plt.subplots(figsize=(8,4))\n"
         "txns[txns.label==0]['amount'].clip(upper=20000).hist(bins=60, alpha=0.6, label='legit', ax=ax, density=True)\n"
         "txns[txns.label==1]['amount'].clip(upper=20000).hist(bins=60, alpha=0.6, label='fraud', ax=ax, density=True)\n"
         "ax.set_xlabel('Amount (INR, clipped at 20k)'); ax.set_ylabel('Density'); ax.legend()\n"
         "ax.set_title('Transaction amount distribution')\n"
         "plt.show()"),
    md("## Fraud rate by payment method"),
    code("txns.groupby('payment_method')['label'].mean().sort_values(ascending=False)"),
    md("## Transactions per day over time (train/val/test boundary visible)"),
    code("daily = txns.set_index('timestamp').resample('D').size()\n"
         "fig, ax = plt.subplots(figsize=(10,3))\n"
         "daily.plot(ax=ax)\n"
         "ax.axvline(pd.Timestamp('2025-09-01'), color='orange', linestyle='--', label='train/val split')\n"
         "ax.axvline(pd.Timestamp('2025-10-01'), color='red', linestyle='--', label='val/test split')\n"
         "ax.set_title('Daily transaction volume'); ax.legend()\n"
         "plt.show()"),
    md("## Scenario breakdown"),
    code("txns.groupby('scenario').agg(count=('transaction_id','count'), avg_amount=('amount','mean')).sort_values('count', ascending=False)"),
])

# ---------------------------------------------------------------------------
# 03: Feature engineering
# ---------------------------------------------------------------------------
build_and_run("03_feature_engineering.ipynb", [
    md("# 03 - Feature Engineering\n\n"
       "Runs `backend/features/engineering.py` on the raw data and inspects the engineered "
       "features: velocity windows, amount deviation, device/IP novelty, and geo deviation."),
    code("import sys\n"
         "sys.path.insert(0, '.')\n"
         "import pandas as pd\n"
         "from backend.features.engineering import load_raw, build_features\n"
         "from pathlib import Path\n\n"
         "txns, customers, devices, ips, merchants = load_raw(Path('data/processed'))\n"
         "df, feature_cols = build_features(txns, customers, devices, ips, merchants)\n"
         "print(f'{len(df):,} rows, {len(feature_cols)} features')\n"
         "feature_cols"),
    md("## Feature summary statistics, split by fraud label"),
    code("df.groupby('label')[feature_cols].mean().T.rename(columns={0: 'legit_mean', 1: 'fraud_mean'})"),
    md("## Velocity features clearly separate the velocity-attack scenario"),
    code("scenario_labels = pd.read_csv('data/processed/transactions_scenario_labels.csv')\n"
         "df2 = df.merge(scenario_labels, on='transaction_id', how='left')\n"
         "df2.groupby('scenario')[['velocity_5min', 'velocity_1hr']].mean().sort_values('velocity_5min', ascending=False)"),
    md("## Device/IP sharing flags clearly separate the abuse-ring scenario"),
    code("df2.groupby('scenario')[['device_shared_flag', 'ip_shared_flag', 'device_customer_count', 'ip_customer_count']].mean()\\\n"
         "    .sort_values('ip_customer_count', ascending=False)"),
    md("## Save engineered feature sample for reference"),
    code("df.head(1000).to_csv('data/processed/features_sample.csv', index=False)\n"
         "print('Saved data/processed/features_sample.csv')"),
])

# ---------------------------------------------------------------------------
# 04: Model training
# ---------------------------------------------------------------------------
build_and_run("04_model_training.ipynb", [
    md("# 04 - Model Training\n\n"
       "Trains the two models used by RiskShield AI on a temporal train/validation split: "
       "the supervised **XGBoost** fraud classifier and the unsupervised **Isolation Forest** "
       "anomaly detector. This mirrors what `run_pipeline.py` does end-to-end; this notebook is "
       "for interactively inspecting the models (feature importance, calibration) in isolation."),
    code("import sys\n"
         "sys.path.insert(0, '.')\n"
         "import pandas as pd\n"
         "from pathlib import Path\n"
         "from backend.features.engineering import load_raw, build_features\n"
         "from backend.models.fraud_models import FraudClassifier, AnomalyDetector\n\n"
         "txns, customers, devices, ips, merchants = load_raw(Path('data/processed'))\n"
         "df, feature_cols = build_features(txns, customers, devices, ips, merchants)\n"
         "for c in feature_cols:\n"
         "    df[c] = df[c].fillna(0)\n\n"
         "train_df = df[df['timestamp'] < '2025-09-01']\n"
         "val_df = df[(df['timestamp'] >= '2025-09-01') & (df['timestamp'] < '2025-10-01')]\n"
         "print(f'Train: {len(train_df):,}  Val: {len(val_df):,}')"),
    md("## Train XGBoost fraud classifier"),
    code("clf = FraudClassifier(feature_cols).fit(train_df, train_df['label'], val_df, val_df['label'])\n"
         "val_score = clf.predict_proba(val_df)\n"
         "from sklearn.metrics import average_precision_score, roc_auc_score\n"
         "print(f'Validation PR-AUC: {average_precision_score(val_df[\"label\"], val_score):.4f}')\n"
         "print(f'Validation ROC-AUC: {roc_auc_score(val_df[\"label\"], val_score):.4f}')"),
    md("## Feature importance"),
    code("import matplotlib\n"
         "matplotlib.use('Agg')\n"
         "%matplotlib inline\n"
         "import matplotlib.pyplot as plt\n"
         "importances = pd.Series(clf.model.feature_importances_, index=feature_cols).sort_values()\n"
         "importances.plot(kind='barh', figsize=(7,5), title='XGBoost feature importance')\n"
         "plt.tight_layout(); plt.show()"),
    md("## Train Isolation Forest anomaly detector"),
    code("anom = AnomalyDetector(feature_cols, contamination=0.05).fit(train_df)\n"
         "anomaly_scores = anom.score(val_df)\n"
         "print(f'Mean anomaly score, legit: {anomaly_scores[val_df.label==0].mean():.3f}')\n"
         "print(f'Mean anomaly score, fraud: {anomaly_scores[val_df.label==1].mean():.3f}')"),
    md("Both models are re-trained inside `run_pipeline.py` as part of the full end-to-end run "
       "(with the model artifacts saved to `evaluation/reports/*.joblib`); this notebook exists "
       "for quick iteration on model choices without re-running the whole pipeline."),
])

# ---------------------------------------------------------------------------
# 05: Evaluation
# ---------------------------------------------------------------------------
build_and_run("05_evaluation.ipynb", [
    md("# 05 - Evaluation\n\n"
       "Loads the results of the full pipeline run (`evaluation/reports/metrics.json`, produced "
       "by `run_pipeline.py`) and walks through the held-out test-set evaluation: classification "
       "metrics, cost-sensitive analysis, and abuse-ring detection quality."),
    code("import json\n"
         "import pandas as pd\n"
         "from pathlib import Path\n\n"
         "metrics = json.loads(Path('evaluation/reports/metrics.json').read_text())\n"
         "metrics['temporal_split']"),
    md("## Classification metrics (held-out October test set)"),
    code("pd.Series(metrics['classification_metrics'])"),
    md("## Cost-sensitive evaluation\n\n"
       "Accuracy is not the headline metric here -- fraud is a 6.5% minority class with highly "
       "asymmetric costs. See `evaluation/metrics.py` for `cost_analysis()`."),
    code("cost = metrics['cost_analysis']\n"
         "baseline = metrics['baseline_ml_only_cost_analysis']\n"
         "comparison = pd.DataFrame({'Fused model': cost, 'ML-only baseline': baseline}).T\n"
         "comparison"),
    code("print(f\"Cost reduction from fusion vs ML-only baseline: {metrics['cost_reduction_vs_ml_only_baseline_pct']}%\")\n"
         "print(f\"Cost-optimal threshold (tuned on validation set): {metrics['optimal_threshold']}\")"),
    md("## Abuse ring detection quality"),
    code("pd.Series(metrics['ring_detection_metrics'])"),
    md("## Evaluation plots\n\n(Precision-recall curve, ROC curve, cost-vs-threshold sweep, risk score distribution)"),
    code("from IPython.display import Image\n"
         "Image(filename='evaluation/reports/evaluation_plots.png')"),
    md("## Sample investigations (agent + policy engine output on top alerts)"),
    code("for ex in metrics['sample_investigations'][:3]:\n"
         "    print(f\"[{ex['risk_level']}] {ex['transaction_id']}  score={ex['risk_score']:.2f}  \"\n"
         "          f\"truth={'FRAUD' if ex['ground_truth_label'] else 'legit'} ({ex['ground_truth_scenario']})\")\n"
         "    print(f\"  -> {ex['investigation_summary']}\")\n"
         "    print(f\"  -> agent recommends: {ex['agent_recommended_action']}, policy sets: {ex['policy_final_action']}\")\n"
         "    print()"),
])

print("\nAll notebooks built and executed successfully.")
