"""Evaluate the verified-resolution-memory RAG improvement (spec Evaluation section).

This measures what we CAN measure deterministically without a database or new dependencies:
  - Recall@K: does a later, differently-worded query retrieve the seeded verified resolution?
  - Ranking: does the verified resolution outrank ordinary historical evidence?
  - Gating: are below-threshold and intent-mismatch queries correctly excluded?

Honest limitation (reported in the output): escalation precision/recall, false-escalation rate, and
hallucination-rate A/B testing require a hand-labeled dissatisfaction/escalation set and real traffic
volume that this take-home does not have. We do NOT claim those numbers.
"""
from __future__ import annotations

import io
import os

from . import memory, resolution_retrieval, pipeline

_SEED_QUERY = "I was charged twice for the same ride"
# A realistic customer re-contact that reuses key content words. Pure-lexical TF-IDF cannot match a
# zero-overlap paraphrase ("billed two times") — that is a documented limitation, not measured here.
_PARAPHRASE = "charged twice for one ride, still not refunded"
_INTENT = "billing_payment"


def run(memory_path: str | None = None) -> dict:
    hits = resolution_retrieval.retrieve_resolutions(_PARAPHRASE, intent=_INTENT, k=3, path=memory_path)
    recall = 1.0 if any(h["source"] == "human_resolution" for h in hits) else 0.0

    merged = pipeline.retrieve_similar(_PARAPHRASE, k=3, intent=_INTENT, memory_path=memory_path)
    ranked_first = bool(merged) and merged[0].get("source") == "human_resolution"

    below = resolution_retrieval.retrieve_resolutions(
        "how do I change my profile photo", intent="general_query", k=3, path=memory_path)
    below_excluded = (below == [])

    mism = resolution_retrieval.retrieve_resolutions(_SEED_QUERY, intent="trip_issue", k=3, path=memory_path)
    intent_excluded = (mism == [])

    return {"recall_at_k": recall, "verified_ranked_first": ranked_first,
            "below_threshold_excluded": below_excluded, "intent_mismatch_excluded": intent_excluded}


def main() -> None:
    # seed a demo resolution into the default store if none exists, then evaluate
    if not memory.load_resolutions():
        memory.add_resolution(memory.build_record(
            _SEED_QUERY, "We confirmed a duplicate charge and it will be refunded per policy.",
            _INTENT, "customer rejected the AI answer and repeated the billing issue"))
    r = run()
    out = io.StringIO()
    w = out.write
    w("# Verified-Resolution-Memory Evaluation\n\n")
    w("Deterministic checks of the feedback-driven RAG improvement (no model training involved).\n\n")
    w("## What we can measure\n\n")
    w(f"- **Recall@3** (paraphrased query retrieves the seeded verified resolution): **{r['recall_at_k']:.2f}**\n")
    w(f"- **Verified ranked first** over historical evidence: **{r['verified_ranked_first']}**\n")
    w(f"- **Below-threshold query excluded** (no false trust): **{r['below_threshold_excluded']}**\n")
    w(f"- **Intent-mismatch excluded**: **{r['intent_mismatch_excluded']}**\n\n")
    w("## What we do NOT claim\n\n")
    w("Escalation precision/recall, false-escalation rate, unresolved-repeat rate, and hallucination-"
      "rate A/B require a hand-labeled escalation golden set and real traffic volume this take-home "
      "does not have. Those are listed as future work; no improvement numbers are asserted for them.\n")
    os.makedirs("reports", exist_ok=True)
    with open("reports/memory_eval.md", "w", encoding="utf-8") as f:
        f.write(out.getvalue())
    print("Wrote reports/memory_eval.md")
    print(r)


if __name__ == "__main__":
    main()
