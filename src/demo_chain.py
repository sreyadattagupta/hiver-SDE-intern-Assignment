"""
Live agent-chain demonstration (CLAUDE.md 0.1 — proves the real pipeline end-to-end).

Runs one message through all four stages and prints exactly what each stage receives and
produces, so the hand-off between stages is visible:

    message -> [1 classify] -> classification -> [2 retrieve] -> grounded cases
            -> [3 draft (uses classification + retrieved)] -> reply
            -> [4 route (uses classification + message)] -> AI/HUMAN decision

Run:  python -m src.demo_chain            (uses whatever provider .env configures; rules if none)
      python -m src.demo_chain --rules    (force the no-network rule path)
"""
from __future__ import annotations

import sys
from . import pipeline, llm

DEMO_MESSAGES = [
    "I was charged twice for one ride and no one is responding. Refund me!",
    "our driver was drunk and driving recklessly, I felt genuinely unsafe",
    "the app keeps crashing every time I try to book",          # the intentional crash-collision bug
    "how do I change the payment method on my account?",
]


def _line(c="─"):
    print(c * 78)


def run_one(message: str, mode: str) -> dict:
    _line("═")
    print(f"INPUT MESSAGE:\n  {message}")
    provider = "rules" if mode == "rules" else (llm.provider_name() or "rules")
    print(f"(pipeline mode: {provider})")

    # STAGE 1 — classify
    _line()
    print("STAGE 1 · classify_intent")
    c = pipeline.classify_intent(message, mode=mode)
    print(f"  → intent={c['intent']}  confidence={c['confidence']}  runner_up={c['runner_up']}")
    print(f"  → method={c['method']}   [this dict is passed to stages 3 and 4]")

    # STAGE 2 — retrieve (input: raw message)
    _line()
    print("STAGE 2 · retrieve_similar   [input: raw message]")
    retrieved = pipeline.retrieve_similar(message, k=3)
    for r in retrieved:
        print(f"  → [{r['pair_id']}] sim={r['score']}  cust={r['customer_msg'][:60]!r}")
    print("  → top-3 pairs passed to stage 3")

    # STAGE 3 — draft (input: message + retrieved from stage 2)
    _line()
    print("STAGE 3 · draft_reply   [input: message + stage-2 retrieved]")
    d = pipeline.draft_reply(message, retrieved)
    print(f"  → draft: {d['draft']}")
    print(f"  → cited={d['cited_example_id']}  method={d['method']}")
    print(f"  → justification: {d['justification'][:120]}")
    if pipeline.is_dm_deflection(d["draft"]):
        print("  ⚠ draft is a 'please DM us' deflection (grounded on paper, low real value)")

    # STAGE 4 — route (input: stage-1 classification + message)
    _line()
    print("STAGE 4 · route_decision   [input: stage-1 classification + message + thread history]")
    rt = pipeline.route_decision(c, message, [])
    print(f"  → DECISION: {rt['decision']}")
    print(f"  → {rt['reason']}")
    _line("═")
    print()
    return {"classification": c, "retrieved": retrieved, "draft": d, "route": rt}


def main():
    mode = "rules" if "--rules" in sys.argv else "auto"
    if mode == "auto" and not llm.llm_available():
        print("(no LLM key found — running the rule-based path; set GROQ_API_KEY to see the LLM path)\n")
    for m in DEMO_MESSAGES:
        run_one(m, mode)


if __name__ == "__main__":
    main()
