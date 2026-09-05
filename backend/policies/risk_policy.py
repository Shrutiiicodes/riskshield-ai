"""
RiskShield AI - Policy / Guardrail Engine

The AI agent may RECOMMEND an action. This engine ENFORCES what is actually
allowed. No autonomous money movement, no irreversible actions. This keeps
the system defense-only and auditable, regardless of what an LLM agent
proposes.
"""

ALLOWED_ACTIONS = {"none", "enhanced_monitoring", "create_review_case", "recommend_hold"}

# Actions an agent might try to request but which are never permitted
BLOCKED_ACTIONS = {
    "block_customer_permanently",
    "transfer_money",
    "refund_money",
    "close_account",
    "auto_ban_device",
}

POLICY_TABLE = {
    "LOW": "none",
    "MEDIUM": "enhanced_monitoring",
    "HIGH": "create_review_case",
    "CRITICAL": "recommend_hold",
}


class PolicyDecision:
    def __init__(self, allowed, action, reason):
        self.allowed = allowed
        self.action = action
        self.reason = reason

    def to_dict(self):
        return {"allowed": self.allowed, "action": self.action, "reason": self.reason}


def evaluate_policy(risk_level: str, agent_requested_action: str = None) -> PolicyDecision:
    """
    The policy engine is the sole authority on final action. The agent's
    requested_action is only ever advisory -- if it asks for something
    outside ALLOWED_ACTIONS (e.g. an irreversible/monetary action), it is
    rejected and the deterministic policy-table action is used instead.
    """
    default_action = POLICY_TABLE.get(risk_level, "enhanced_monitoring")

    if agent_requested_action in BLOCKED_ACTIONS:
        return PolicyDecision(
            allowed=False,
            action=default_action,
            reason=f"Requested action '{agent_requested_action}' is not permitted under policy. "
                   f"Falling back to risk-tier default: '{default_action}'.",
        )

    if agent_requested_action and agent_requested_action not in ALLOWED_ACTIONS:
        return PolicyDecision(
            allowed=False,
            action=default_action,
            reason=f"Unrecognized action '{agent_requested_action}'. Using risk-tier default: '{default_action}'.",
        )

    return PolicyDecision(allowed=True, action=default_action, reason=f"Risk tier '{risk_level}' maps to '{default_action}'.")
