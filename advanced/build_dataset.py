"""
Fine-tuning dataset builder (Phases 7 & 8).

Produces a leakage-safe, stratified train/val set for the DistilBERT intent classifier:

  source        : data/uber_pairs.csv  (1,500 grounding pairs) — DISJOINT from the golden set,
                  which is sampled from data/uber_golden_pool.csv. The golden set is NEVER used
                  for training (it is the held-out test set).
  labels        : WEAK labels from the rule classifier (pipeline._classify_rules). Honest caveat:
                  the student can inherit the teacher-rules' mistakes; the real test is golden-set
                  F1 vs the rules baseline (advanced/train_classifier.py measures exactly that).
  augmentation  : a small set of hand-written HARD NEGATIVES (Phase 8) — minimal pairs that teach
                  context ("app crashed" vs "driver crashed", "ride cancelled" vs "driver
                  cancelled", "charged twice" vs "one pending"). These are NOT from golden.
  cleaning      : preprocess.clean_for_embedding + near-duplicate dedup.
  split         : stratified 85/15 train/val, class distribution preserved, seed=42.

Run:  python -m advanced.build_dataset
"""
from __future__ import annotations

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import _classify_rules, INTENTS          # noqa: E402
from src.preprocess import clean_for_embedding             # noqa: E402

PAIRS = "data/uber_pairs.csv"
OUT_DIR = "advanced/data"

# Phase 8 — curated hard negatives (context-teaching minimal pairs). Hand-labeled, not from golden.
HARD_NEGATIVES = [
    ("the app keeps crashing every time I open it", "service_complaint"),
    ("my driver crashed the car into another vehicle", "safety_incident"),
    ("website crashed while I was booking", "service_complaint"),
    ("our uber was in a crash on the highway", "safety_incident"),
    ("my ride was cancelled by the app automatically", "trip_issue"),
    ("the driver cancelled on me after I waited 10 minutes", "trip_issue"),
    ("I was charged twice for the same trip", "billing_payment"),
    ("I see two transactions but one is only pending", "billing_payment"),
    ("I feel unsafe, the driver was speeding and reckless", "safety_incident"),
    ("the app says my connection is unsafe", "account_access"),
    ("the driver was rude and unprofessional to me", "trip_issue"),
    ("the driver was dangerous and nearly hit a pedestrian", "safety_incident"),
    ("cannot log in after reinstalling the app", "account_access"),
    ("my account was hacked and someone took rides", "account_access"),
    ("my food never arrived from uber eats", "delivery_order"),
    ("uber eats charged me but cancelled the order", "billing_payment"),
    ("how do I add a second stop to my trip", "general_query"),
    ("where is my refund, it has been a week with no reply", "service_complaint"),
    ("driver took a longer route and overcharged me", "billing_payment"),
    ("promo code is not applying at checkout", "billing_payment"),
    ("male driver kept making inappropriate comments", "safety_incident"),
    ("is there a phone number I can call for support", "general_query"),
    ("my driver assaulted me at the drop off point", "safety_incident"),
    ("there was no accident, the app just froze", "service_complaint"),
]


def _dedup(df, col):
    seen, keep = set(), []
    for _, r in df.iterrows():
        key = r[col][:80]
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
    return pd.DataFrame(keep)


def main():
    df = pd.read_csv(PAIRS)
    df["text"] = df["customer_msg"].fillna("").map(clean_for_embedding)
    df = df[df["text"].str.len() >= 5]
    df["label"] = df["customer_msg"].map(lambda m: _classify_rules(m)["intent"])
    df = _dedup(df, "text")

    hard = pd.DataFrame(
        [{"text": clean_for_embedding(t), "label": lab} for t, lab in HARD_NEGATIVES])
    data = pd.concat([df[["text", "label"]], hard], ignore_index=True)
    data = data[data["label"].isin(INTENTS)].reset_index(drop=True)

    tr, va = train_test_split(data, test_size=0.15, random_state=42, stratify=data["label"])
    os.makedirs(OUT_DIR, exist_ok=True)
    tr.to_csv(f"{OUT_DIR}/train.csv", index=False)
    va.to_csv(f"{OUT_DIR}/val.csv", index=False)

    print(f"train={len(tr)}  val={len(va)}  (+{len(hard)} hard negatives)")
    print("\nlabel distribution (train):")
    print(tr["label"].value_counts().to_string())
    print(f"\nWrote {OUT_DIR}/train.csv and val.csv")
    print("NOTE: golden_set.csv is NOT in this data (held-out test set) — no leakage.")


if __name__ == "__main__":
    main()
