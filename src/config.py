"""Tunable constants for the verified-resolution-memory RAG loop.

Every weight/threshold used by satisfaction detection, escalation, memory, and resolution
retrieval lives here so it is discoverable, justifiable, and testable (spec Steps 5/11/12).
Defaults were chosen conservatively (favour precision — avoid false escalation and avoid trusting
weak matches) and their effect is measured in src/memory_eval.py.
"""
from __future__ import annotations

import os

# --- resolution retrieval (Step 10/11) ---
# A verified resolution is only eligible as evidence if its similarity to the incoming message is
# at least this. Calibrated empirically against the combined (query + resolution) TF-IDF embedding:
# cross-topic/irrelevant queries score ~0.0, on-topic re-contacts score ~0.30-0.64 (see
# reports/memory_eval.md). 0.30 cleanly separates the two. NOTE: pure-lexical TF-IDF has a
# paraphrase ceiling — a re-contact with zero shared content words ("billed two times" vs "charged
# twice") still misses; this is a documented limitation (sentence-transformers is sandbox-gated).
RESOLUTION_SIM_THRESHOLD = 0.30
# Additive score bump applied to a verified resolution at ranking time so it outranks an equally
# relevant ordinary historical reply — but NOT a much-more-relevant one (bump is small on purpose).
VERIFIED_TRUST_BOOST = 0.15

# --- conflict detection (Step 13) ---
# Two verified resolutions conflict when their queries are this similar (same problem) ...
CONFLICT_SIM_THRESHOLD = 0.75
# ... but their resolution TEXT similarity is below this (materially different answers).
RESOLUTION_TEXT_DIFF_THRESHOLD = 0.6

# --- satisfaction / escalation (Steps 3/4) ---
# Implicit-dissatisfaction score at/above which a message is treated as dissatisfied.
DISSATISFACTION_THRESHOLD = 0.6
# Token-Jaccard similarity between the current message and a prior customer message above which we
# count it as a repeated/unresolved question.
REPEAT_SIM_THRESHOLD = 0.5
# Policy: a single 👎 does not force escalation; a SECOND 👎 does (spec-approved).
ESCALATE_SECOND_THUMBS_DOWN = True

# --- persistence (Step 8/9) ---
MEMORY_PATH = os.path.join("data", "resolution_memory.jsonl")

# --- shared conversation store + feedback (human-in-the-loop backend) ---
# One JSON file per conversation lives here; both the customer app and the admin dashboard read/write
# it, so escalations raised in one browser session are visible to a human agent in another (the
# Streamlit equivalent of a shared backend — no HTTP server needed).
CONVERSATIONS_DIR = os.path.join("data", "conversations")
# Raw customer feedback (👍/👎) is logged here. This is the RAW tier of the learning pipeline:
# raw feedback -> (validation) -> only HUMAN-VERIFIED resolutions become trusted knowledge in
# MEMORY_PATH. Helpful clicks never auto-enter the vector/RAG memory.
FEEDBACK_LOG_PATH = os.path.join("data", "feedback_log.jsonl")
