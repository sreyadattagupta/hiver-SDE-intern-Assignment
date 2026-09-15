# Design — Customer Satisfaction → Human Escalation → Verified Resolution Memory → RAG Improvement

Date: 2026-09-15
Feature owner spec for extending the existing **Uber_Support AI Agent** (Hiver SDE take-home).

## 1. Goal

Extend the existing pipeline with a production-minded, human-in-the-loop feedback loop:

```
customer msg → AI response → satisfaction signal
  satisfied   → continue (AI-handled)
  unsatisfied → escalate → human agent → verified resolution
              → resolution memory → embedding → future RAG retrieval → better grounded responses
```

The system learns **only from human-verified, closed resolutions** — never from raw AI output,
raw thumbs-down, or unverified user statements. This is **retrieval augmentation from verified
resolutions**, *not* RLHF / online model training. Terminology is enforced in code, tests, docs.

## 2. What already exists (reuse, do not replace)

- `src/pipeline.py`: `classify_intent`, `retrieve_similar`, `draft_reply`, `verify_draft`,
  `detect_safety`, `route_decision`. Dual-path (LLM-with-fallback → rules), honest `method` field.
- `src/retrieval.py`: TF-IDF / LSA / hybrid / gated `st` over `data/uber_pairs.csv` (1500 pairs),
  `functools.lru_cache`'d sklearn indexes. Contract: `[{pair_id, customer_msg, brand_reply, score, method}]`.
- `src/llm.py`: multi-provider fallback chain, `last_provider()` honesty.
- `app.py`: thin Streamlit viewer — 4 live stage cards (left, own scroll) + chat (right, own scroll),
  `st.session_state.chat`.
- `src/eval.py`, `tests/test_pipeline.py`, `tests/test_advanced.py`.

**No vector DB, no external embedding service. We keep it that way.**

## 3. New modules (each: one purpose, well-defined interface, independently testable)

### `src/satisfaction.py` — dissatisfaction detection
```
detect_dissatisfaction(message, history=None, explicit_feedback=None, mode="auto")
  -> {"dissatisfied": bool, "score": float 0..1, "signals": [str], "method": str}
```
- **Explicit**: `explicit_feedback` ∈ {`helpful`, `not_helpful`, `human_request`, None}.
- **Implicit** (rules path, dependency-free, mirrors `detect_safety` style):
  - whole-word phrase patterns ("not what I asked", "isn't helping", "already told you",
    "still doesn't work", "keep saying the same", "talk to a human", "real person",
    "answer is wrong") with negation guard.
  - **repeat detection**: token-Jaccard similarity between current msg and prior customer msgs
    in `history` above a threshold → "repeated unresolved question" signal.
  - **frustration ≠ technical complaint**: a bare technical term ("app keeps crashing") does NOT
    count; dissatisfaction requires a rejection/complaint-about-help pattern OR explicit feedback
    OR a repeat. This reuses the same context-discipline as `detect_safety`.
- **LLM path** (when key present): a strict-JSON classifier that reads current msg + previous AI
  reply + short history and returns `{dissatisfied, score, reason}`. Honest `method` = `llm:<provider>`.

### `src/escalation.py` — transparent decision engine
```
decide(classification, message, history, feedback, draft_meta) 
  -> {"route": "AI"|"HUMAN", "reason": str, "confidence": float, "signals": [str]}
```
Signals aggregated (superset of existing `route_decision`):
`explicit_human_request`, `thumbs_down`, `high_frustration`, `repeated_unresolved`,
`safety_incident`, `low_classifier_confidence`, `low_retrieval_confidence`,
`groundedness_failure`, `llm_failure`, `repeated_failed_ai`.
- Reuses `pipeline.detect_safety`, classifier confidence, retrieval top-score, `verify_draft` result.
- **Policy (approved)**: single 👎 does **not** force escalation — it adds a strong signal and
  surfaces the "Talk to Human" prompt; HUMAN triggers on: explicit human request, safety, 2nd 👎,
  👎+repeat, high dissatisfaction score ≥ threshold, groundedness hard-fail, or LLM total failure.
- `reason` is concise, user-safe (no hidden chain-of-thought). `confidence` = normalized signal strength.
- `route_decision` kept unchanged for back-compat; `decide` is the richer entry point used by the UI.

### `src/memory.py` — verified resolution store
```
add_resolution(record) -> {"status": "added"|"duplicate"|"rejected_unverified"|"conflict", "resolution_id": str}
load_resolutions() -> [record...]
find_conflicts(record, existing=None) -> [conflicting_record...]
```
- Persistence: `data/resolution_memory.jsonl` (one JSON record per line).
- **Record schema** (Step 8):
  `{resolution_id, original_query, ai_response, human_resolution, intent, escalation_reason,
    human_verified: true, created_at, source: "human_resolution", conflict_flag: bool}`.
- `resolution_id` = deterministic sha1 of normalized (original_query + "␟" + human_resolution)[:16].
- **Gate (Step 12)**: reject unless `human_verified is True` and record is closed. → `rejected_unverified`.
- **Dedup**: same `resolution_id` already present → `duplicate`, no re-insert.
- **Conflict (Step 13)**: a stored record with same `intent` and query-similarity ≥ conflict threshold
  but a *materially different* resolution → mark new record `conflict_flag=True`, keep both, do NOT
  overwrite. Retrieval prefers most-recent verified where a conflict exists (documented).

### `src/resolution_retrieval.py` — verified-memory retrieval layer
```
retrieve_resolutions(query, intent=None, k=3) -> [{pair_id, customer_msg, brand_reply, score, source, verified}...]
```
- Embeds each stored resolution's `original_query` with the **already-fitted** TF-IDF vectorizer
  and LSA SVD from `src/retrieval.py` via `transform()` (same vector space — scores comparable).
  Cached; invalidated when the jsonl mtime changes.
- Gating before a resolution is eligible: `verified is True`, `score ≥ RESOLUTION_SIM_THRESHOLD`,
  and (if `intent` given) intent compatible. Maps to the retrieval contract with
  `customer_msg = original_query`, `brand_reply = human_resolution`, `source="human_resolution"`.

## 4. Wiring into existing pipeline (additive, back-compat)

- `pipeline.retrieve_similar(message, k, method, intent=None, use_memory=True)`:
  1. `resolution_retrieval.retrieve_resolutions(message, intent)` (verified layer)
  2. existing `retrieval.retrieve(message, k, method)` (historical layer)
  3. **evidence ranking**: merge, apply `VERIFIED_TRUST_BOOST` to verified items' scores, sort, take k.
  - When jsonl is empty/missing → identical to current behavior → **existing tests unchanged**.
  - Returned items carry `source` ("human_resolution"|"historical") and `verified` flags so the UI and
    draft justification can explain *why* a verified resolution was used.
- `draft_reply` unchanged in contract; it already grounds on retrieved evidence, so a verified
  resolution simply becomes higher-trust evidence. Justification will name the verified source.

## 5. Config — `src/config.py` (constants, justified + tested)

```
RESOLUTION_SIM_THRESHOLD = 0.45   # below → not relevant enough to trust as verified answer
VERIFIED_TRUST_BOOST     = 0.15   # additive score bump for verified evidence at ranking time
CONFLICT_SIM_THRESHOLD   = 0.75   # same-intent query similarity above which differing resolutions conflict
DISSATISFACTION_THRESHOLD= 0.6    # implicit score at/above which we treat as dissatisfied
ESCALATE_SECOND_THUMBS_DOWN = True
```
Effect of each weight is measured in `memory_eval.py` and asserted in tests (threshold boundary cases).

## 6. UI (`app.py`) — additive

- **Feedback controls** after each AI reply: `👍 Helpful` / `👎 Not Helpful` / `💬 Talk to Human`
  as real `st.button`s that mutate `session_state` (turn-level feedback + escalation state).
- **Conversation memory**: multi-turn thread persists in `session_state.chat` (already present) with
  Customer / AI / Feedback / Escalation rows.
- **Human Agent Console panel** (in-app, appears when a turn is `ESCALATED`): shows the structured
  escalation record (conversation id, original msg, history, intent, AI response, retrieved evidence,
  safety status, escalation reason, timestamp, status) + a resolution form
  (final resolution, optional category, optional internal note, "Resolve & Verify" button).
  On submit → `memory.add_resolution(human_verified=True)`.
- **Feedback-loop visualization strip** after a verified resolution:
  `AI RESPONSE → 👎 FEEDBACK → 🔴 ESCALATION → 👤 RESOLUTION → ✓ VERIFIED → 🧠 MEMORY → 🔎 FUTURE RETRIEVAL`.
- **Live pipeline** stage cards keep real running/complete states (no faked statuses).
- **Layout invariants** (already satisfied, must be verified): left Agent Run panel and right chat
  each in their own `st.container(height=...)` → independent scrollbars; header (custom topbar) not
  clipped (Streamlit header hidden); chat gets the larger column (ratio ~1 : 2.7).
- Sensitive-data hygiene: internal notes and metadata rendered in the human console only, kept out
  of the customer-visible chat; no secrets/keys logged.

## 7. Evaluation — `src/memory_eval.py`

- Seed a known verified resolution; issue a later semantically-related query; assert it is retrieved
  (Recall@K) and ranked above historical evidence; show below-threshold / intent-mismatch queries do
  **not** retrieve it.
- Before-vs-after groundedness/relevance on a small scripted scenario set (deterministic, rules path).
- **Honesty**: explicitly state that escalation precision/recall and hallucination-rate A/B require a
  hand-labeled dissatisfaction/escalation set and real traffic volume we don't have; report what we
  *can* measure and mark the rest as future work. No unsupported improvement claims.

## 8. Tests — `tests/test_feedback_loop.py` (Step 18, 32 cases)

Explicit (👍/👎/human), implicit ("not what I asked", "isn't helping", repeat, "want a human",
normal technical complaint = NOT dissatisfied), escalation (safety, low confidence, groundedness
fail, LLM fail), memory (add, store, embed, insert, dedup, reject-unverified, retrieve-later,
irrelevant-not-retrieved), RAG (historical, verified, conflict, below-threshold, intent mismatch),
plus back-compat assertions that existing pipeline contracts/tests still hold. UI-state logic tested
at the function level (feedback mutates state, escalation status, resolution flow); scroll/header are
manual acceptance items noted in the validation report.

## 9. Failure analysis (documented in reports/)

incorrect escalation · missed dissatisfaction · false frustration · irrelevant human resolution ·
conflicting resolutions · stale resolution · duplicate memory · bad embedding · retrieval-threshold
failure · human resolution with unsupported info — each with Root Cause / Impact / Mitigation /
Remaining Limitation.

## 10. Non-goals (YAGNI)

No Pinecone/Qdrant/Chroma/OpenAI-embeddings/ST-required/Postgres/Redis. No model fine-tuning in
this loop. No auth system. No multi-user concurrency guarantees (single-process file append is enough
for the take-home; documented as a limitation).

## 11. Terminology guardrail

Code, tests, README, ARCHITECTURE use: "human-verified resolution memory",
"retrieval augmentation from resolved escalations", "feedback-driven RAG improvement".
Never "RLHF" / "self-training" / "the model retrains itself".
