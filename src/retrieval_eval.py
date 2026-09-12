"""
Retrieval quality comparison (Phase 4C / Phase 21): tfidf vs lsa vs st vs hybrid.

We have no human relevance judgments for retrieval, so we use a transparent PROXY:
  intent-match@k = fraction of golden queries whose top-k retrieved historical pairs contain at
  least one pair whose (rule-classified) intent equals the query's TRUE golden intent.
Higher = the retriever surfaces topically-on-target evidence more often. Also report mean top-1
similarity and mean latency/query. This is a proxy, labeled as such — not a human eval.

Run:  python -m src.retrieval_eval
"""
from __future__ import annotations

import time
import json
import os
import pandas as pd

from . import retrieval
from .pipeline import _classify_rules

GOLDEN = "data/golden_set.csv"
OUT = "reports/retrieval_eval.md"
K = 3


def _intent_of(msg: str) -> str:
    return _classify_rules(msg)["intent"]


def eval_method(method: str, gold: pd.DataFrame):
    hits, sims, t0 = 0, [], time.time()
    for _, row in gold.iterrows():
        res = retrieval.retrieve(row["customer_msg"], k=K, method=method)
        sims.append(res[0]["score"] if res else 0.0)
        got = {_intent_of(r["customer_msg"]) for r in res}
        if row["true_intent"] in got:
            hits += 1
    dt = (time.time() - t0) / max(1, len(gold))
    return {"intent_match@%d" % K: hits / len(gold), "mean_top1_sim": sum(sims) / len(sims),
            "ms_per_query": dt * 1000}


def main():
    gold = pd.read_csv(GOLDEN)
    methods = ["tfidf", "lsa", "hybrid"]
    if retrieval.st_available():
        methods.append("st")
    rows = {m: eval_method(m, gold) for m in methods}

    os.makedirs("reports", exist_ok=True)
    lines = ["# Retrieval comparison (proxy metric)\n",
             f"Golden queries: {len(gold)} · k={K} · metric = intent-match@{K} "
             "(top-k contains a pair whose rule-intent == query's true intent).\n",
             f"sentence-transformers available in this env: **{retrieval.st_available()}**\n",
             "| method | intent-match@%d | mean top-1 sim | ms/query |" % K,
             "|---|---|---|---|"]
    best = max(rows, key=lambda m: rows[m]["intent_match@%d" % K])
    for m, r in rows.items():
        star = " ⭐" if m == best else ""
        lines.append(f"| {m}{star} | {r['intent_match@%d'%K]:.3f} | {r['mean_top1_sim']:.3f} | {r['ms_per_query']:.1f} |")
    lines += ["", f"**Winner by proxy: `{best}`.** Note: `st` (neural) similarity is on a different "
              "scale than tfidf/lsa cosine, so compare methods by intent-match, not raw similarity.",
              "", "_Proxy caveat: intent-match uses the rule classifier to label retrieved pairs, so "
              "it measures topical alignment, not human-judged usefulness._"]
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT}")
    json.dump(rows, open("reports/retrieval_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
