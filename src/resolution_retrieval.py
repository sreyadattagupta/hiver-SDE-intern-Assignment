"""Retrieval over human-verified resolution memory (spec Step 10).

Each stored resolution's ORIGINAL customer query is embedded in the SAME fitted TF-IDF space as
the 1500-pair historical corpus (via retrieval.tfidf_cosine), so a verified-resolution score is
directly comparable to a historical score and both can be merged/ranked together in the pipeline.

Gating (Step 10/12): a verified resolution is eligible only if it is verified, its query-similarity
to the incoming message clears RESOLUTION_SIM_THRESHOLD, and (when the caller supplies an intent)
the intent matches. High similarity alone is never sufficient — the threshold + intent + verified
flag together decide eligibility.
"""
from __future__ import annotations

from . import config, memory, retrieval


def retrieve_resolutions(query: str, intent: str | None = None, k: int = 3, path: str | None = None) -> list:
    records = memory.load_resolutions(path)
    if not records:
        return []
    # Embed the COMBINED (original query + human resolution) per spec Step 9: this widens the token
    # overlap available to the lexical retriever, so a re-contact that reuses words from either the
    # problem OR the resolution can match.
    docs = [f"{r['original_query']} {r['human_resolution']}" for r in records]
    scores = retrieval.tfidf_cosine(query, docs)
    out = []
    for rec, score in zip(records, scores):
        if rec.get("human_verified") is not True:
            continue
        if intent is not None and rec.get("intent") != intent:
            continue
        s = float(score)
        if s < config.RESOLUTION_SIM_THRESHOLD:
            continue
        out.append({
            "pair_id": f"res_{rec['resolution_id']}",
            "customer_msg": rec["original_query"],
            "brand_reply": rec["human_resolution"],
            "score": round(s, 4),
            "method": "resolution_memory",
            "source": "human_resolution",
            "verified": True,
            "intent": rec.get("intent", ""),
            "conflict_flag": rec.get("conflict_flag", False),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:k]
