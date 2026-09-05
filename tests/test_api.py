import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analytics_summary(client):
    r = client.get("/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    assert "precision" in body and "recall" in body


def test_alerts_filter_by_risk(client):
    r = client.get("/risk/alerts", params={"risk": "CRITICAL", "limit": 5})
    assert r.status_code == 200
    alerts = r.json()
    assert all(a["risk_level"] == "CRITICAL" for a in alerts)


def test_investigate_unknown_transaction_404(client):
    r = client.get("/risk/investigate/TXN_does_not_exist")
    assert r.status_code == 404


def test_live_scoring(client):
    r = client.post("/transactions/score", json={
        "amount": 20000, "amount_dev_ratio": 9.0, "device_is_new": 1,
        "ip_shared_flag": 1, "velocity_5min": 8,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_case_lifecycle(client):
    txn_id = client.get("/risk/alerts", params={"risk": "CRITICAL", "limit": 1}).json()[0]["transaction_id"]
    create_resp = client.post("/cases/", json={
        "transaction_id": txn_id, "risk_score": 0.95, "risk_level": "CRITICAL",
        "investigation_summary": "test summary",
        "agent_recommended_action": "recommend_hold",
        "policy_final_action": "recommend_hold",
    })
    assert create_resp.status_code == 200
    case_id = create_resp.json()["case_id"]
    assert create_resp.json()["status"] == "open"

    resolve_resp = client.post(f"/cases/{case_id}/resolve", json={
        "analyst_decision": "confirmed_fraud", "notes": "matches known ring"
    })
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["status"] == "closed"
