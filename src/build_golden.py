"""
Step 2 (CLAUDE.md build order): golden evaluation set.

Attaches hand-assigned true_intent labels to the 200 held-out messages sampled in
data/_golden_sample.csv and writes data/golden_set.csv.

Labeling protocol (documented for the report):
  - 200 messages sampled (seed=7) from the HELD-OUT pool (disjoint from the retrieval corpus).
  - Each message read and labeled by hand into one of the 7 intents (see src/pipeline.py).
  - `ambiguous=True` marks messages with genuinely mixed or context-dependent intent
    (e.g. "driver cancelled AND charged me" -> billing vs trip). These are kept in the set
    but reported separately, because headline accuracy on ambiguous items is not meaningful.
  - Labeling rule of thumb for mixed billing/trip cases: if the customer's concrete ASK is a
    refund / charge reversal, label billing_payment; if it's about the ride/driver experience
    itself, label trip_issue. For UberEats: food quality/missing -> delivery_order; refund ask
    -> billing_payment.

Run:  python -m src.build_golden
"""
from __future__ import annotations

import os
import pandas as pd

POOL = os.path.join("data", "uber_golden_pool.csv")
SAMPLE = os.path.join("data", "_golden_sample.csv")
OUT = os.path.join("data", "golden_set.csv")
SAMPLE_SEED = 7   # MUST match the seed the LABELS below were assigned against

# hand labels, index-aligned to data/_golden_sample.csv (seed=7); 20 rows x 10 = 200
LABELS = [
    "general_query","delivery_order","trip_issue","billing_payment","trip_issue","general_query","general_query","trip_issue","billing_payment","general_query",
    "general_query","billing_payment","service_complaint","billing_payment","general_query","trip_issue","trip_issue","general_query","safety_incident","general_query",
    "safety_incident","billing_payment","delivery_order","general_query","delivery_order","account_access","trip_issue","billing_payment","trip_issue","billing_payment",
    "general_query","general_query","service_complaint","general_query","trip_issue","general_query","account_access","service_complaint","service_complaint","safety_incident",
    "trip_issue","general_query","billing_payment","service_complaint","billing_payment","billing_payment","billing_payment","general_query","service_complaint","safety_incident",
    "account_access","delivery_order","general_query","service_complaint","service_complaint","delivery_order","delivery_order","trip_issue","service_complaint","general_query",
    "service_complaint","billing_payment","service_complaint","billing_payment","safety_incident","delivery_order","trip_issue","account_access","trip_issue","safety_incident",
    "trip_issue","account_access","billing_payment","billing_payment","general_query","trip_issue","general_query","billing_payment","service_complaint","delivery_order",
    "billing_payment","trip_issue","general_query","account_access","billing_payment","billing_payment","general_query","account_access","general_query","safety_incident",
    "account_access","billing_payment","general_query","delivery_order","delivery_order","trip_issue","account_access","delivery_order","billing_payment","billing_payment",
    "trip_issue","billing_payment","service_complaint","trip_issue","delivery_order","billing_payment","billing_payment","general_query","billing_payment","billing_payment",
    "trip_issue","service_complaint","safety_incident","service_complaint","service_complaint","billing_payment","account_access","account_access","billing_payment","safety_incident",
    "billing_payment","billing_payment","general_query","safety_incident","trip_issue","service_complaint","billing_payment","general_query","trip_issue","billing_payment",
    "billing_payment","trip_issue","service_complaint","trip_issue","service_complaint","general_query","billing_payment","trip_issue","delivery_order","billing_payment",
    "billing_payment","billing_payment","delivery_order","account_access","service_complaint","general_query","billing_payment","general_query","general_query","safety_incident",
    "billing_payment","general_query","trip_issue","account_access","general_query","service_complaint","billing_payment","billing_payment","delivery_order","billing_payment",
    "trip_issue","service_complaint","trip_issue","service_complaint","billing_payment","general_query","general_query","service_complaint","service_complaint","delivery_order",
    "billing_payment","billing_payment","service_complaint","billing_payment","service_complaint","general_query","account_access","general_query","general_query","billing_payment",
    "billing_payment","billing_payment","account_access","trip_issue","trip_issue","trip_issue","general_query","billing_payment","billing_payment","general_query",
    "general_query","billing_payment","billing_payment","trip_issue","service_complaint","service_complaint","service_complaint","service_complaint","billing_payment","general_query",
]

# indices with genuinely mixed / context-dependent intent (kept, reported separately)
AMBIGUOUS = {0,6,8,9,11,12,13,22,24,27,28,30,44,46,58,61,64,73,80,85,86,87,99,106,109,112,
             115,119,128,136,140,145,146,150,155,156,157,162,167,181,184,195,196,198}


def main() -> None:
    # regenerate the exact labeled sample from the held-out pool (deterministic seed) so a fresh
    # clone reproduces it — no dependence on a pre-existing scratch file.
    if os.path.exists(SAMPLE):
        df = pd.read_csv(SAMPLE)
    else:
        df = pd.read_csv(POOL).sample(n=len(LABELS), random_state=SAMPLE_SEED).reset_index(drop=True)
        df.to_csv(SAMPLE, index=False)
    assert len(df) == len(LABELS), f"sample has {len(df)} rows but {len(LABELS)} labels"
    df = df[["pair_id", "customer_msg"]].copy()
    df["true_intent"] = LABELS
    df["ambiguous"] = [i in AMBIGUOUS for i in range(len(df))]
    df["notes"] = ["mixed/context-dependent" if a else "" for a in df["ambiguous"]]
    df.to_csv(OUT, index=False)

    print(f"Wrote {len(df)} labeled examples -> {OUT}")
    print(f"  ambiguous/mixed: {df['ambiguous'].sum()}  ({df['ambiguous'].mean()*100:.0f}%)")
    print("\nlabel distribution:")
    print(df["true_intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
