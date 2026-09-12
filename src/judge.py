"""
LLM-as-judge for reply quality + judge-vs-human agreement (CLAUDE.md 7).

Grades drafted replies on three binary dimensions — grounded, helpful, polite — and measures
how well the LLM judge agrees with hand ratings (Cohen's kappa + % agreement per dimension).

To keep this reproducible WITHOUT an API key, the drafts judged are the deterministic
rules-mode drafts (each = the closest historical brand reply), captured in
data/_judge_sample.csv, and hand-rated once in data/human_ratings.csv. When a key is present,
the LLM judge grades those exact same 25 drafts, so the agreement numbers are apples-to-apples.

Run:  python -m src.judge      (needs an LLM key; see src/llm.py)
"""
from __future__ import annotations

import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import llm
from .pipeline import _loads

HUMAN = os.path.join("data", "human_ratings.csv")
OUT = os.path.join("reports", "judge_agreement.md")
DIMS = ["grounded", "helpful", "polite"]


def _judge_one(customer_msg: str, draft: str) -> dict:
    system = (
        "You are a strict QA reviewer for customer-support replies. Grade the draft reply to the "
        "customer message on three BINARY dimensions (1 = yes, 0 = no):\n"
        "grounded: is the reply topically appropriate to the customer's actual issue (not a wrong-"
        "topic or wrong-language response)?\n"
        "helpful: does it move toward resolving THIS specific issue, rather than being a generic "
        "'please DM us' deflection with no substance?\n"
        "polite: is the tone courteous and professional?\n"
        'Respond ONLY with JSON: {"grounded":0|1,"helpful":0|1,"polite":0|1}.'
    )
    raw = llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f'Customer: "{customer_msg}"\nDraft reply: "{draft}"'}],
        temperature=0.0, max_tokens=800, json_mode=True,
    )
    d = _loads(raw)
    return {dim: int(bool(d.get(dim, 0))) for dim in DIMS}


def main() -> None:
    if not llm.llm_available():
        print("No LLM key set — cannot run the LLM judge. Set GROQ_API_KEY (free) in .env.")
        print("Human ratings are ready in data/human_ratings.csv; agreement fills in once a key exists.")
        return

    df = pd.read_csv(HUMAN)
    print(f"Judging {len(df)} drafts with LLM provider '{llm.provider_name()}'...")
    jg = {dim: [] for dim in DIMS}
    for _, row in df.iterrows():
        grades = _judge_one(row["customer_msg"], row["draft"])
        for dim in DIMS:
            jg[dim].append(grades[dim])

    lines = ["# LLM-Judge vs Human Agreement\n",
             f"Provider (last used): **{llm.last_provider()}**  |  Drafts judged: **{len(df)}** "
             "(deterministic rules-mode drafts)\n",
             "| Dimension | % agreement | Cohen's kappa | human mean | judge mean |",
             "|---|---|---|---|---|"]
    for dim in DIMS:
        human = df[f"h_{dim}"].tolist()
        judge = jg[dim]
        agree = sum(int(a == b) for a, b in zip(human, judge)) / len(human)
        try:
            kappa = cohen_kappa_score(human, judge)
        except Exception:
            kappa = float("nan")
        kstr = f"{kappa:.3f}" if kappa == kappa else "undefined (no variance)"
        lines.append(f"| {dim} | {agree*100:.0f}% | {kstr} | "
                     f"{sum(human)/len(human):.2f} | {sum(judge)/len(judge):.2f} |")

    lines += [
        "",
        "**Reading this honestly:** 'polite' has near-zero variance (Uber's templated replies are "
        "always courteous), so its kappa is meaningless — high agreement there is not evidence the "
        "judge is trustworthy. The dimensions that matter are *grounded* and especially *helpful*, "
        "where the judge must distinguish a real resolution from a polite 'please DM us' deflection. "
        "Trust the judge only as far as its kappa on *helpful* holds up.",
    ]
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
