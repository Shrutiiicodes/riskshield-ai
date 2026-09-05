"""
RiskShield AI - Investigation Agent

Design principle: ML determines risk; the agent REASONS OVER EVIDENCE to
produce a human-readable investigation summary and a bounded recommendation.
It never decides the final action -- that's the Policy Engine's job
(see backend/policies/risk_policy.py).

Two modes are supported:
  1. Template-based reasoning (default, no external dependency, fully
     deterministic and offline -- used by run_pipeline.py and tests).
  2. LLM-backed reasoning: if an ANTHROPIC_API_KEY is available, swap in
     `llm_investigate()` which sends the same structured evidence to Claude
     and asks it to produce the narrative. The tool-call surface
     (get_transaction, get_customer_history, etc.) is identical either way,
     so this is a drop-in upgrade, not a redesign.
"""
import os
import json


TOOLS = [
    "get_transaction", "get_customer_history", "get_device_history",
    "get_ip_history", "get_related_transactions", "get_risk_features",
    "get_model_explanation", "get_cluster_information",
    "create_case", "recommend_action",
]

# NOTE: block_customer(), transfer_money(), refund_money() are deliberately
# absent from TOOLS. The agent has no capability to move money or take
# irreversible action -- see policies/risk_policy.py for enforcement.


def _describe_reason(feature_name, value):
    labels = {
        "velocity_5min": lambda v: f"{int(v)} transactions in the last 5 minutes",
        "velocity_1hr": lambda v: f"{int(v)} transactions in the last hour",
        "amount_dev_ratio": lambda v: f"amount is {v:.1f}x this customer's historical average",
        "device_is_new": lambda v: "transaction originates from a newly observed device",
        "device_shared_flag": lambda v: "device is shared across multiple customer accounts",
        "ip_shared_flag": lambda v: "IP address is shared across multiple customer accounts",
        "is_foreign_country": lambda v: "transaction location differs from the customer's home country",
        "failed_txn_count_24h": lambda v: f"{int(v)} failed transaction attempts in the last 24 hours",
        "historical_chargebacks": lambda v: f"{int(v)} prior chargebacks on this account",
        "seconds_since_prev_txn": lambda v: f"only {int(v)}s since the customer's previous transaction",
    }
    fn = labels.get(feature_name)
    return fn(value) if fn else f"{feature_name} = {value}"


def build_evidence_reasons(row, shap_contributions, top_k=5):
    """Rank SHAP contributions and translate the top ones into plain language."""
    sorted_feats = sorted(shap_contributions.items(), key=lambda kv: -abs(kv[1]))[:top_k]
    reasons = []
    for feat, contrib in sorted_feats:
        if contrib <= 0:
            continue
        val = row.get(feat, None)
        if val is None:
            continue
        reasons.append({"feature": feat, "contribution": round(float(contrib), 4),
                         "description": _describe_reason(feat, val)})
    return reasons


def template_investigate(evidence: dict) -> dict:
    """
    Deterministic, template-based investigation summary. This is the
    'agent' in offline/CI mode -- it consumes the same structured evidence
    an LLM-backed version would, so swapping in a real model later requires
    no change to callers.
    """
    txn = evidence["transaction"]
    reasons = evidence["reasons"]
    cluster = evidence.get("cluster")
    risk_score = evidence["risk_score"]
    risk_level = evidence["risk_level"]

    reason_phrases = [r["description"] for r in reasons]
    narrative_bits = []
    if reason_phrases:
        narrative_bits.append(
            f"This transaction (₹{txn['amount']:.0f}) was flagged with a risk score of "
            f"{risk_score:.2f} ({risk_level}). Key contributing signals: " +
            "; ".join(reason_phrases[:3]) + "."
        )
    else:
        narrative_bits.append(
            f"This transaction (₹{txn['amount']:.0f}) received a risk score of {risk_score:.2f} "
            f"({risk_level}) based on the fused model output."
        )

    if cluster and cluster.get("cluster_size", 1) >= 3:
        narrative_bits.append(
            f"The associated device/IP is part of a network of {cluster['cluster_size']} accounts, "
            f"consistent with coordinated abuse rather than an isolated incident."
        )

    if evidence.get("related_transaction_count", 0) > 1:
        narrative_bits.append(
            f"{evidence['related_transaction_count']} related transactions were found for this "
            f"customer/device/IP cluster within the recent window."
        )

    narrative = " ".join(narrative_bits)

    # bounded recommendation -- must be one of the allowed policy actions
    if risk_level == "CRITICAL":
        recommended_action = "recommend_hold"
    elif risk_level == "HIGH":
        recommended_action = "create_review_case"
    elif risk_level == "MEDIUM":
        recommended_action = "enhanced_monitoring"
    else:
        recommended_action = "none"

    confidence = min(0.99, 0.5 + risk_score / 2)

    return {
        "summary": narrative,
        "evidence_reasons": reasons,
        "recommended_action": recommended_action,
        "confidence": round(confidence, 2),
        "mode": "template",
    }


def llm_investigate(evidence: dict) -> dict:
    """
    Optional LLM-backed investigation. Requires ANTHROPIC_API_KEY in the
    environment. Falls back to template_investigate() on any failure
    (e.g. no network, no key, unavailable service) -- see Case 2 in the
    failure-handling design: "LLM unavailable -> ML scoring + rule-based
    recommendation continue".
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return template_investigate(evidence)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "You are a fraud investigation assistant for a payments company. "
            "You are given structured evidence about a flagged transaction. "
            "Write a concise (3-4 sentence) investigation summary explaining why it "
            "was flagged, in plain language, based ONLY on the evidence given. "
            "Then recommend exactly one action from this fixed list: "
            "none, enhanced_monitoring, create_review_case, recommend_hold. "
            "You may NOT recommend blocking, transferring, or refunding money -- "
            "those actions do not exist in this system. "
            "Respond as JSON: {\"summary\": ..., \"recommended_action\": ..., \"confidence\": 0-1}.\n\n"
            f"Evidence:\n{json.dumps(evidence, default=str, indent=2)}"
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        parsed = json.loads(text.strip().strip("`").replace("json\n", "", 1))
        parsed["evidence_reasons"] = evidence["reasons"]
        parsed["mode"] = "llm"
        # re-validate the action against policy allowlist even though we asked nicely
        from backend.policies.risk_policy import ALLOWED_ACTIONS
        if parsed.get("recommended_action") not in ALLOWED_ACTIONS:
            parsed["recommended_action"] = template_investigate(evidence)["recommended_action"]
        return parsed
    except Exception:
        return template_investigate(evidence)


def investigate(evidence: dict) -> dict:
    """Public entry point used by run_pipeline.py / backend API."""
    return llm_investigate(evidence)
