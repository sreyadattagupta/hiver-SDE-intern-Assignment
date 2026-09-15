# Verified-Resolution-Memory Evaluation

Deterministic checks of the feedback-driven RAG improvement (no model training involved).

## What we can measure

- **Recall@3** (paraphrased query retrieves the seeded verified resolution): **1.00**
- **Verified ranked first** over historical evidence: **True**
- **Below-threshold query excluded** (no false trust): **True**
- **Intent-mismatch excluded**: **True**

## What we do NOT claim

Escalation precision/recall, false-escalation rate, unresolved-repeat rate, and hallucination-rate A/B require a hand-labeled escalation golden set and real traffic volume this take-home does not have. Those are listed as future work; no improvement numbers are asserted for them.
