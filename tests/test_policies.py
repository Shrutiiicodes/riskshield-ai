import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.policies.risk_policy import evaluate_policy, ALLOWED_ACTIONS, BLOCKED_ACTIONS


def test_policy_table_defaults():
    assert evaluate_policy("LOW").action == "none"
    assert evaluate_policy("MEDIUM").action == "enhanced_monitoring"
    assert evaluate_policy("HIGH").action == "create_review_case"
    assert evaluate_policy("CRITICAL").action == "recommend_hold"


def test_policy_rejects_blocked_actions():
    for action in BLOCKED_ACTIONS:
        decision = evaluate_policy("CRITICAL", action)
        assert decision.allowed is False
        assert decision.action == "recommend_hold"  # falls back to safe default


def test_policy_allows_permitted_agent_action():
    decision = evaluate_policy("HIGH", "create_review_case")
    assert decision.allowed is True
    assert decision.action == "create_review_case"


def test_no_overlap_between_allowed_and_blocked():
    assert ALLOWED_ACTIONS.isdisjoint(BLOCKED_ACTIONS)
