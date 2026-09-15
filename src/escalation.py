"""Transparent AI-vs-HUMAN escalation decision engine (spec Step 4).

A superset of pipeline.route_decision: it folds in explicit feedback and implicit dissatisfaction
on top of the existing safety / confidence / frustration / repeat signals, plus groundedness and
LLM-failure signals from the draft stage. The reason is concise and user-safe (no chain-of-thought):
it names the signals that fired, nothing more.
"""
from __future__ import annotations

from . import config, pipeline, satisfaction

# user-safe phrasing per signal (shown to reviewers / logged)
_REASON = {
    "explicit_human_request": "the customer explicitly requested a human agent",
    "safety_incident": "this is a physical-safety incident",
    "second_thumbs_down": "the customer marked the answer unhelpful more than once",
    "high_dissatisfaction": "the customer rejected the previous response and remained dissatisfied",
    "repeated_unresolved_question": "the customer repeated an unresolved issue",
    "low_classifier_confidence": "the request was too ambiguous to classify confidently",
    "low_retrieval_confidence": "no sufficiently relevant past case was found",
    "groundedness_failure": "the drafted answer failed grounding checks",
    "llm_failure": "the AI generation service was unavailable",
}


def decide(classification: dict, message: str, history=None, feedback=None, draft_meta=None) -> dict:
    feedback = feedback or {}
    draft_meta = draft_meta or {}
    signals = []

    intent = classification.get("intent", "")
    conf = float(classification.get("confidence", 0.0))

    # safety (context-aware; reuse the existing detector)
    if pipeline.detect_safety(message)["is_safety"] or intent == "safety_incident":
        signals.append("safety_incident")

    # explicit + implicit dissatisfaction
    explicit = feedback.get("explicit")
    if explicit == "human_request":
        signals.append("explicit_human_request")
    td = int(feedback.get("thumbs_down_count", 0))
    if config.ESCALATE_SECOND_THUMBS_DOWN and td >= 2:
        signals.append("second_thumbs_down")

    # Dissatisfaction is a REACTION to a prior answer — only meaningful once there is conversational
    # context (a prior turn in history) or an explicit reaction. On a first-contact message we do NOT
    # treat "I have a problem / I want a refund" as dissatisfaction (that would false-escalate almost
    # every complaint). An explicit human request in the raw text is always honoured, though.
    if any(r.search(message) for r in satisfaction._HUMAN_RE) and "explicit_human_request" not in signals:
        signals.append("explicit_human_request")
    has_context = bool(history) or explicit is not None
    if has_context:
        dis = satisfaction.detect_dissatisfaction(message, history=history, explicit_feedback=explicit)
        if "explicit_human_request" in dis["signals"] and "explicit_human_request" not in signals:
            signals.append("explicit_human_request")
        if "repeated_unresolved_question" in dis["signals"]:
            signals.append("repeated_unresolved_question")
        if dis["dissatisfied"] and dis["score"] >= 0.7 and explicit != "not_helpful":
            signals.append("high_dissatisfaction")

    # confidence + retrieval + groundedness + llm health
    if conf < 0.55:
        signals.append("low_classifier_confidence")
    top = draft_meta.get("retrieval_top_score")
    if top is not None and float(top) < 0.10:
        signals.append("low_retrieval_confidence")
    verif = draft_meta.get("verification")
    if verif is not None and verif.get("ok") is False:
        signals.append("groundedness_failure")
    if draft_meta.get("llm_failed"):
        signals.append("llm_failure")

    # de-dup, preserve order
    seen, uniq = set(), []
    for s in signals:
        if s not in seen:
            seen.add(s); uniq.append(s)
    signals = uniq

    route = "HUMAN" if signals else "AI"
    if route == "HUMAN":
        clauses = [_REASON.get(s, s) for s in signals]
        reason = "Escalated because " + "; and ".join(clauses) + "."
        confidence = round(min(1.0, 0.5 + 0.15 * len(signals)), 3)
    else:
        reason = (f"Handled by AI: confident ({conf:.2f}) non-safety intent '{intent}', "
                  "no dissatisfaction or grounding issues.")
        confidence = round(conf, 3)
    return {"route": route, "reason": reason, "confidence": confidence, "signals": signals}
