"""
RiskShield AI - Case Endpoints

Implements the `create_case()` tool referenced in the investigation agent's
allowed tool surface (backend/agents/investigation_agent.py). This is an
in-memory store for demo purposes -- swap for a real table (Postgres, per
the original design doc) in production; the interface would stay the same.

  POST /cases/            -- open a new review case for a transaction
  GET  /cases/            -- list open/closed cases
  GET  /cases/{case_id}   -- fetch one case
  POST /cases/{case_id}/resolve -- record an analyst's final decision (audit trail)
"""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/cases", tags=["cases"])

_CASES = {}  # case_id -> case dict (in-memory; resets on restart)


class CreateCaseRequest(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    investigation_summary: str
    agent_recommended_action: str
    policy_final_action: str


class ResolveCaseRequest(BaseModel):
    analyst_decision: str  # e.g. "confirmed_fraud", "false_positive", "escalated"
    notes: str = ""


@router.post("/")
def create_case(req: CreateCaseRequest):
    case_id = f"CASE_{uuid.uuid4().hex[:8]}"
    case = {
        "case_id": case_id,
        "transaction_id": req.transaction_id,
        "risk_score": req.risk_score,
        "risk_level": req.risk_level,
        "investigation_summary": req.investigation_summary,
        "agent_recommended_action": req.agent_recommended_action,
        "policy_final_action": req.policy_final_action,
        "status": "open",
        "analyst_decision": None,
        "notes": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_at": None,
    }
    _CASES[case_id] = case
    return case


@router.get("/")
def list_cases(status: str = None):
    cases = list(_CASES.values())
    if status:
        cases = [c for c in cases if c["status"] == status]
    return sorted(cases, key=lambda c: c["created_at"], reverse=True)


@router.get("/{case_id}")
def get_case(case_id: str):
    case = _CASES.get(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    return case


@router.post("/{case_id}/resolve")
def resolve_case(case_id: str, req: ResolveCaseRequest):
    case = _CASES.get(case_id)
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    case["status"] = "closed"
    case["analyst_decision"] = req.analyst_decision
    case["notes"] = req.notes
    case["resolved_at"] = datetime.now(timezone.utc).isoformat()
    return case
