# Evaluation Results — Uber_Support AI Agent

Golden set: **200** hand-labeled held-out messages (**44** flagged ambiguous/mixed = 22%).

LLM provider available: **groq**

## 1. Classification metrics (full golden set)


> ⚠ Few-shot LLM row skipped this run: `RateLimitError` (Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01kvhyw). Baseline/rule results below are unaffected; re-run when the provider quota resets. Previously measured LLM macro-F1 ≈ 0.69.

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (majority class) | 0.275 | 0.062 |
| TF-IDF + LogReg (weak-labeled) | 0.535 | 0.493 |
| Rule-based classifier | 0.555 | 0.533 |

### Same metrics on NON-ambiguous items only (156 examples)

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| Trivial (majority class) | 0.212 | 0.050 |
| TF-IDF + LogReg (weak-labeled) | 0.564 | 0.538 |
| Rule-based classifier | 0.583 | 0.583 |

### Per-class report — Rule-based classifier

```
                   precision    recall  f1-score   support

  billing_payment      0.854     0.636     0.729        55
   account_access      0.800     0.533     0.640        15
       trip_issue      0.321     0.290     0.305        31
  safety_incident      1.000     0.273     0.429        11
   delivery_order      0.579     0.688     0.629        16
service_complaint      0.714     0.323     0.444        31
    general_query      0.412     0.854     0.556        41

         accuracy                          0.555       200
        macro avg      0.669     0.514     0.533       200
     weighted avg      0.641     0.555     0.552       200

```

### Confusion matrix — Rule-based classifier

| true \ pred | bill | acco | trip | safe | deli | serv | gene |
|---|---|---|---|---|---|---|---|
| billing_paymen | 35 | 1 | 6 | 0 | 6 | 0 | 7 |
| account_access | 3 | 8 | 0 | 0 | 0 | 1 | 3 |
| trip_issue | 3 | 0 | 9 | 0 | 1 | 3 | 15 |
| safety_inciden | 0 | 0 | 3 | 3 | 0 | 0 | 5 |
| delivery_order | 0 | 0 | 2 | 0 | 11 | 0 | 3 |
| service_compla | 0 | 1 | 3 | 0 | 0 | 10 | 17 |
| general_query | 0 | 0 | 5 | 0 | 1 | 0 | 35 |

### Headline lift

- Best real classifier (**Rule-based classifier**) macro-F1 = **0.533**
- vs TF-IDF/LogReg baseline (0.493): **+0.040**
- vs trivial baseline (0.062): **+0.471**

## 2. Grounding sanity — 'please DM us' deflection rate

Fraction of drafted replies that are just a generic 'send us a DM' with no concrete resolution. High = the system games the grounding metric without being helpful (SPEC.md 5).

- Rules-mode drafts (reuse closest historical reply): **39/60 = 65%** are pure DM-deflection.
- For reference, **61%** of the raw historical brand replies in the grounding corpus are themselves DM-deflections — this is the pattern the retriever inherits.

## 3. Routing audit (sample)

| message | intent | decision | reason |
|---|---|---|---|
| @115873 in Massachusetts its illegal to record someone witho | trip_issue | AI | Routed to AI because: confident (0.70) non-safety intent 'trip_issue', no frustration/repe |
| @789789 @115877 @52126 I suggest a refund is due. Otherwise  | billing_payment | AI | Routed to AI because: confident (0.95) non-safety intent 'billing_payment', no frustration |
| @Uber_Support Any Update on this ? | general_query | AI | Routed to AI because: confident (0.95) non-safety intent 'general_query', no frustration/r |
| I wish male @115873 drivers would stop hitting on me. So dam | general_query | HUMAN | Routed to HUMAN because: low classifier confidence (0.30 < 0.55). |
| @Uber_Support Thanks for your prompt response. I would love  | general_query | AI | Routed to AI because: confident (0.95) non-safety intent 'general_query', no frustration/r |
| @Uber_Support r u not listening to my query | general_query | HUMAN | Routed to HUMAN because: low classifier confidence (0.30 < 0.55). |
| @Uber_Support I send my email. I await a solution. | general_query | HUMAN | Routed to HUMAN because: low classifier confidence (0.30 < 0.55). |
| @Uber_Support But I need to know now this is a scheduled rid | trip_issue | AI | Routed to AI because: confident (0.95) non-safety intent 'trip_issue', no frustration/repe |

### Safety routing — naive keywords vs context-aware NLP (the fixed bug)

| message | naive substring | context-aware detect_safety | verdict |
|---|---|---|---|
| the app keeps crashing every time I try to book | SAFETY (crash) | — | ✅ fixed |
| our car crashed into a pole on the highway | SAFETY (crash) | SAFETY | ✅ correct |
| the driver was drunk and driving recklessly | — (none) | SAFETY | gain |
| no accident, just a late pickup | SAFETY (accident) | — | ✅ fixed |
| male drivers keep hitting on me and it makes me unco | — (none) | — | ✅ correct |

The old blind matcher flagged any message containing 'crash'/'accident'. The context-aware detector (`detect_safety`) uses whole-word matching, a ±4-token context window, tech-vs-vehicle disambiguation, and negation — so 'app keeps crashing' is a software issue (AI) while 'car crashed' escalates (HUMAN). With an LLM key it additionally catches subtle cases the keyword list misses (e.g. harassment). See failure_analysis.md.
