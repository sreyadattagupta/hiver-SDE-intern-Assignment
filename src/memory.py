"""Human-verified resolution memory (spec Steps 6/8/12/13).

This is NOT model training. A human support agent closes an escalated ticket with a final,
verified resolution; that resolution is appended here as trusted knowledge for future RAG
retrieval. Only human_verified + closed records may enter. Duplicates are prevented by a
content hash; conflicting resolutions for the same problem are flagged and BOTH kept (never
silently overwritten) so the conflict is auditable.

Storage: data/resolution_memory.jsonl — one JSON record per line (deterministic, git-diffable,
no database dependency).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from difflib import SequenceMatcher

from . import config


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def resolution_id(original_query: str, human_resolution: str) -> str:
    """Deterministic id from the (query, resolution) content — used for dedup."""
    raw = _norm(original_query) + "␟" + _norm(human_resolution)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def load_resolutions(path: str | None = None) -> list:
    path = path or config.MEMORY_PATH
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def find_conflicts(record: dict, existing: list | None = None) -> list:
    """Records with the SAME intent and a very similar query but a materially DIFFERENT
    resolution text (spec Step 13). Query-similarity and resolution-difference both use difflib
    ratio so this module needs no sklearn dependency."""
    existing = existing if existing is not None else load_resolutions()
    conflicts = []
    for e in existing:
        if e.get("intent") != record.get("intent"):
            continue
        q_sim = _ratio(record["original_query"], e["original_query"])
        r_sim = _ratio(record["human_resolution"], e["human_resolution"])
        if q_sim >= config.CONFLICT_SIM_THRESHOLD and r_sim < config.RESOLUTION_TEXT_DIFF_THRESHOLD:
            conflicts.append(e)
    return conflicts


def build_record(original_query: str, human_resolution: str, intent: str, escalation_reason: str,
                 ai_response: str = "", category: str = "", internal_note: str = "") -> dict:
    """Assemble a full verified-resolution record (human_verified=True, closed)."""
    rid = resolution_id(original_query, human_resolution)
    rec = {
        "resolution_id": rid,
        "original_query": original_query,
        "ai_response": ai_response,
        "human_resolution": human_resolution,
        "intent": intent,
        "escalation_reason": escalation_reason,
        "category": category,
        "internal_note": internal_note,
        "human_verified": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "human_resolution",
        "conflict_flag": False,
    }
    return rec


def add_resolution(record: dict, path: str | None = None) -> dict:
    """Append a record IF it passes the trust gate + dedup. Returns status.
    Gate: must be human_verified True. Dedup: same resolution_id already present.
    Conflict: flagged on the new record (both kept)."""
    path = path or config.MEMORY_PATH
    if record.get("human_verified") is not True:
        return {"status": "rejected_unverified", "resolution_id": record.get("resolution_id", ""),
                "conflict_flag": False}
    existing = load_resolutions(path)
    rid = record.get("resolution_id") or resolution_id(record["original_query"], record["human_resolution"])
    record["resolution_id"] = rid
    if any(e["resolution_id"] == rid for e in existing):
        return {"status": "duplicate", "resolution_id": rid, "conflict_flag": False}
    conflicts = find_conflicts(record, existing)
    record["conflict_flag"] = bool(conflicts)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "added", "resolution_id": rid, "conflict_flag": bool(conflicts)}
