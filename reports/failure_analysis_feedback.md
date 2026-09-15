# Failure Analysis — Feedback / Verified-Memory Loop

Each row: Failure · Root Cause · Impact · Mitigation · Remaining Limitation.

| Failure | Root Cause | Impact | Mitigation | Remaining Limitation |
|---|---|---|---|---|
| Incorrect escalation | Rule pattern over-fires on borderline language | Human queue noise | 2-signal / 2nd-👎 policy, precision-first thresholds | Rare borderline phrasing still slips through |
| Missed dissatisfaction | Novel phrasing not in patterns | Unhappy customer stays with AI | LLM verifier path when key present; content-overlap repeat detection | Rules-only mode misses subtle sarcasm |
| False frustration | Technical term read as anger | Needless escalation | Bare technical complaint explicitly NOT dissatisfaction (test-enforced) | Mixed messages (tech + mild annoyance) ambiguous |
| Irrelevant human resolution retrieved | Similarity high but topic drift | Wrong guidance surfaced | `RESOLUTION_SIM_THRESHOLD` + intent gate + verified flag | Threshold is a global constant, not per-intent |
| Conflicting resolutions | Two verified answers for the same problem | Ambiguous evidence | Conflict flag on ingest, both kept, never overwritten | No human review queue for flagged conflicts yet |
| Stale resolution | Policy changed after resolution stored | Outdated guidance surfaced | `created_at` retained; most-recent preferred where a conflict exists | No TTL / expiry implemented |
| Duplicate memory | Same resolution re-submitted | Storage bloat / double-weighting | Deterministic content-hash (`resolution_id`) dedup | Near-duplicate (reworded) not deduped |
| Bad embedding | Query text too short / all stopwords | Weak retrieval score, missed match | Combined (query+resolution) embedding widens tokens; threshold filters weak hits | TF-IDF misses zero-overlap paraphrase (ST gated in sandbox) |
| Retrieval threshold failure | Threshold too low → noise, too high → misses | Over/under-trust of memory | Threshold calibrated to measured scores (irrelevant≈0.0, on-topic 0.30–0.64); boundary-tested | Single global value, not learned |
| Human resolution with unsupported info | Agent writes an unverifiable promise | Bad knowledge enters memory | Human is the verification authority; internal note kept separate from customer-visible text | No automated fact-check of human-entered text |

## The documented paraphrase ceiling

The verified-memory retriever uses the same dependency-free TF-IDF space as the historical
retriever (sentence-transformers weights are CDN-blocked in this sandbox — see `src/retrieval.py`).
Measured behaviour of the combined (query + resolution) embedding:

| Incoming re-contact | Similarity to seeded billing resolution | Retrieved? (threshold 0.30) |
|---|---|---|
| "charged twice for one ride, still no refund" | 0.64 | ✅ yes |
| "duplicate charge on my ride, refund please" | 0.32 | ✅ yes |
| "billed **two times** for one trip" (zero content-word overlap) | 0.00 | ❌ no — lexical ceiling |
| "how do I change my profile photo" | 0.00 | ❌ correctly excluded |

A customer who re-contacts reusing key content words is matched; a full synonym paraphrase with no
shared content word is not. This is an honest limitation of lexical retrieval, not a bug — the fix is
neural embeddings (already gated in `retrieval.py`, used automatically when the model is available).
