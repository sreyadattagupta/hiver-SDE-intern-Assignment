# Failure Analysis & the Misleading Headline Number

All numbers below come from `python -m src.eval` on the 200-example golden set (rule-based
path, no API key). Re-run it to reproduce.

---

## The mandatory section: what's misleading about my headline number

**Headline temptation:** "The agent retrieves a real historical Uber reply for every message
and drafts a grounded, on-brand response." True — and almost meaningless. Here is why.

### 1. Grounding is real but the ground truth is lazy — the "please DM us" trap
- **61%** of the raw historical brand replies in the grounding corpus are pure
  *"send us a DM / note"* deflections with no concrete resolution.
- So a system that grounds its draft in the closest historical reply inherits that: **65%**
  of rules-mode drafts are themselves DM-deflections.
- A groundedness or similarity metric therefore rewards the model for reproducing a template
  that resolves nothing. **A high "grounded" score is not evidence of a useful agent.** The
  honest metric is *helpful* (does it resolve THIS issue?), which is far lower — and which the
  demo UI flags with a ⚠ whenever a draft is a deflection.

### 2. Naive safety keywords were imprecise both ways — now fixed with context-aware NLP
The original router flagged safety by blind substring matching, which failed in both directions.
That has been replaced by `pipeline.detect_safety()` (an LLM verifier with an NLP rule fallback:
whole-word matching + ±4-token context window + tech-vs-vehicle disambiguation + negation). The
before/after is in `reports/eval_results.md`:

| message | naive substring | context-aware | outcome |
|---|---|---|---|
| "the app keeps crashing" | SAFETY (`crash`) | not safety | **false positive fixed** → routes AI |
| "our car crashed into a pole" | SAFETY | SAFETY | still correct → HUMAN |
| "the driver was drunk and reckless" | not flagged | SAFETY | **false negative fixed** → HUMAN |
| "no accident, just a late pickup" | SAFETY (`accident`) | not safety | **negation handled** → AI |
| "male drivers keep hitting on me" | not flagged | SAFETY (LLM) | **subtle harassment caught** → HUMAN |

- Honest caveat: the **rule-only** fallback still can't catch keyword-free harassment
  ("male drivers keep hitting on me") — only the **LLM verifier** does. So safety recall is
  classifier-dependent (rules < LLM), and any single "safety coverage" number is meaningless
  unless it states which path produced it.
- Trade-off measured: making the *rule classifier* context-aware nudged its macro-F1 from 0.545
  to 0.533 — it now declines a few lucky substring safety-hits in exchange for far fewer false
  escalations. The LLM path (the real system, macro-F1 ≈ 0.69) is unaffected.

### 3. The rule-vs-TF-IDF gap is fake; only the LLM lift is real
- Macro-F1: rule-based **0.545** vs TF-IDF/LogReg **0.533** (**+0.012**).
- This +0.012 is not a real win: the TF-IDF model was **trained on the rule classifier's own
  weak labels**, so it is a distillation of the rules. Two classifiers agreeing when one taught
  the other tells us nothing about correctness on edge cases they both get wrong.
- The few-shot **LLM** (Groq `openai/gpt-oss-120b`) reaches **0.690** — a genuine **+0.157**
  over TF-IDF, evaluated on the same human labels. But note *where* it wins: `safety_incident`
  recall goes **0.36 → 1.00** and `service_complaint` recall **0.32 → 0.90**, while
  `general_query` recall *drops* to **0.34** (the LLM reclassifies many vague follow-ups as
  service_complaint — see confusion matrix). So the headline +0.157 is not uniform improvement;
  it's a large safety/complaint gain partly offset by worse general_query handling.

### 4. The golden set is skewed toward easy, unambiguous tickets
- **22%** of sampled messages are genuinely ambiguous/mixed (e.g. *"driver cancelled AND
  charged me"* — billing or trip?). We report metrics on the full set AND the non-ambiguous
  subset (0.545 → 0.595 macro-F1). The headline number quietly benefits from every ambiguous
  item we happened to resolve the "expected" way.
- Distribution is billing-heavy (55/200) mirroring real Uber traffic, so accuracy is dominated
  by billing performance; a model that only nailed billing would post a respectable-looking
  accuracy while failing minority intents.

---

## Top 5 failure modes (from the confusion matrices)

1. **general_query collapses under the LLM (recall 0.34).** The LLM reclassifies vague
   follow-ups/rants ("your services are going down the drain", "any update on this?") as
   `service_complaint` — 20 general_query → service_complaint. This is the LLM's single biggest
   error and the mirror image of the rule classifier's failure, which instead dumped
   trip_issue/service_complaint *into* general_query (14 and 17 cases). Neither model handles the
   content-free catch-all well; they just fail in opposite directions.
2. **trip_issue is weak in both (F1 rules 0.305 / LLM 0.525).** Terse ride complaints
   ("waiting outside 40 mins", "no music no AC 1 star") get pulled toward safety or service.
3. **safety detection is classifier-dependent (after the context-aware fix).** Rule-path recall
   **0.36** (misses harassment/privacy
   with no explicit keyword) vs LLM recall **1.00**. If anyone reports a single "safety coverage"
   number without saying which classifier produced it, it is meaningless — they differ 3x.
4. **billing vs delivery confusion.** UberEats refund requests ("order rejected but I was
   charged") straddle both; billing↔delivery leaks in both classifiers. Genuinely ambiguous.
5. **DM-deflection drafts (65%).** Even when classification is right, the drafted reply is
   often a content-free "please DM us", so a correct intent does not yield a useful answer.

---

## Judge-vs-human agreement (measured — do not trust the judge blindly)
`python -m src.judge` grades 25 deterministic drafts and compares to hand ratings
(`reports/judge_agreement.md`):

| Dimension | % agreement | Cohen's kappa | human mean | judge mean |
|---|---|---|---|---|
| grounded | 48% | **0.085** | 0.84 | 0.40 |
| helpful | 60–64% | **~0.22–0.30** | 0.52 | 0.20–0.24 |
| polite | 96% | 0.000 (no variance) | 1.00 | 0.96 |

(Judge grades vary slightly run-to-run even at temperature 0, so `helpful` kappa lands in a
~0.22–0.30 band across runs — still only *fair*, which is the point.)

This is the credibility check, and it partly *fails*:
- **`grounded` kappa 0.085 ≈ chance.** The judge and I mean different things by "grounded": I
  scored topical appropriateness (0.84 pass), the judge treated DM-deflections as not grounded
  (0.40 pass). A "groundedness" quality score is therefore **not trustworthy** — it depends
  entirely on whose definition you use.
- **`helpful` kappa 0.295 (weak/fair).** The judge is systematically *stricter* than the human
  (0.24 vs 0.52), which happens to align with reality (most drafts are deflections), but the low
  kappa means we cannot lean on the judge's per-item helpful calls.
- **`polite` 96% agreement is meaningless** — every templated reply is polite, so there is no
  variance (kappa 0). Any headline "quality score" that averages in politeness is inflated by a
  dimension the model cannot fail.

**Takeaway:** report the LLM classification F1 (which is checked against hand labels), but treat
the LLM judge's reply-quality scores as indicative at best until agreement improves.

---

## What I'd do with one more week
- Few-shot LLM classifier as the primary path (interface already built), evaluated on these
  same human labels — the only honest way to beat the rule/TF-IDF tie.
- Replace TF-IDF retrieval with `sentence-transformers` embeddings for semantic matching.
- Filter DM-deflection replies OUT of the grounding corpus, or down-weight them, so drafts are
  grounded in cases that actually resolved something.
- ~~Split the `crash` keyword by context~~ — **done**: `detect_safety()` now disambiguates
  car-crash vs app-crash with a context window + negation + an LLM verifier (see §2).
- Expand the golden set and add a second independent labeler to measure human-human agreement
  as the ceiling for judge-human agreement.
