"""Pure, Streamlit-free state transitions for the support console UI (spec Steps 2/5/14).

Kept separate from app.py so the feedback -> escalation state machine and the escalation-record
builder are unit-testable without a running Streamlit server. app.py imports these and only owns
widget wiring / rendering.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import config, escalation


def apply_feedback(turns: list, turn_id: int, feedback: str, message: str = "", history=None) -> dict:
    """Apply explicit feedback to the AI turn `turn_id`. Returns
    {turns, thumbs_down_count, escalated, escalation_signals}.
    Policy (config.ESCALATE_SECOND_THUMBS_DOWN): 1st 👎 records only; explicit human request or a
    2nd 👎 escalates. Uses escalation.decide so the UI shares the engine's policy."""
    turns = [dict(t) for t in turns]
    target = next((t for t in turns if t.get("turn_id") == turn_id and t.get("role") == "ai"), None)
    if target is None:
        return {"turns": turns, "thumbs_down_count": 0, "escalated": False, "escalation_signals": []}
    target["feedback"] = feedback
    thumbs_down_count = sum(1 for t in turns if t.get("role") == "ai" and t.get("feedback") == "not_helpful")

    decision = escalation.decide(
        {"intent": target.get("intent", ""), "confidence": target.get("confidence", 0.9)},
        message or "", history=history or [],
        feedback={"explicit": feedback, "thumbs_down_count": thumbs_down_count})
    escalated = decision["route"] == "HUMAN"
    if escalated:
        target["escalated"] = True
        target["escalation_reason"] = decision["reason"]
    return {"turns": turns, "thumbs_down_count": thumbs_down_count,
            "escalated": escalated, "escalation_signals": decision["signals"]}


def build_escalation_record(conversation_id, original_message, history, intent, ai_response,
                            retrieved, safety_status, reason) -> dict:
    """Structured escalation record given to the human agent (spec Step 5)."""
    return {
        "conversation_id": conversation_id,
        "original_message": original_message,
        "history": list(history or []),
        "intent": intent,
        "ai_response": ai_response,
        "retrieved_evidence": [{"pair_id": r.get("pair_id"), "score": r.get("score")} for r in (retrieved or [])],
        "safety_status": safety_status,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ESCALATED",
    }
