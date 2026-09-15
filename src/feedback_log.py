"""Raw customer-feedback log — the first tier of the learning pipeline (spec Step 4/16).

Tiers, kept deliberately distinct so nothing untrusted leaks into knowledge:

    RAW feedback (this file, data/feedback_log.jsonl)
        -> VALIDATION (human review / quality check)
        -> TRUSTED knowledge = ONLY human-verified resolutions (src/memory.py)

A 👍 or 👎 is recorded here for analytics and to trigger escalation — it NEVER auto-enters the
RAG/resolution memory. Only a human agent closing a ticket with a verified resolution produces
trusted knowledge. This module has no dependency on Streamlit and is unit-testable.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from . import config


def log_feedback(conversation_id: str, message_id: str, feedback: str, customer_query: str = "",
                 ai_response: str = "", intent: str = "", path: str | None = None) -> dict:
    """Append one raw feedback event. Returns the stored record."""
    path = path or config.FEEDBACK_LOG_PATH
    rec = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "feedback": feedback,               # helpful | not_helpful | human_request
        "customer_query": customer_query,
        "ai_response": ai_response,
        "intent": intent,
        "validated": False,                 # raw tier — not yet reviewed
        "trusted": False,                   # never trusted here; only verified resolutions are
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_feedback(path: str | None = None) -> list:
    path = path or config.FEEDBACK_LOG_PATH
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def feedback_stats(path: str | None = None) -> dict:
    """Small aggregate for the admin/learning view."""
    recs = load_feedback(path)
    stats = {"total": len(recs), "helpful": 0, "not_helpful": 0, "human_request": 0}
    for r in recs:
        stats[r.get("feedback", "")] = stats.get(r.get("feedback", ""), 0) + 1
    return stats
