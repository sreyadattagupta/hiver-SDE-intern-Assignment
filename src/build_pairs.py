"""
Step 1 (CLAUDE.md build order): thread reconstruction.

Reads the raw twcs.csv, filters to Uber_Support, walks the reply chains into
(customer inbound message -> brand's resolving reply) pairs, and writes:

  data/uber_pairs.csv        grounding corpus (retrieval draws from this)
  data/uber_golden_pool.csv  HELD-OUT customer messages for golden-set labeling
                             (disjoint from uber_pairs.csv, so eval never tests on
                             messages the retriever has memorized)

Run:  python -m src.build_pairs
"""
from __future__ import annotations

import os
import pandas as pd

RAW = os.path.join("data", "twitter_support", "twcs.csv")
BRAND = "Uber_Support"
PAIRS_OUT = os.path.join("data", "uber_pairs.csv")
POOL_OUT = os.path.join("data", "uber_golden_pool.csv")

N_GROUNDING = 1500     # size of grounding corpus
N_POOL = 600           # held-out customer msgs to sample the golden set from
SEED = 42


def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # strip leading @handle mentions and collapse whitespace; keep the rest as-is (messy on purpose)
    return " ".join(text.split()).strip()


def main() -> None:
    if not os.path.exists(RAW):
        raise FileNotFoundError(f"Raw dataset not found at {RAW}")

    print(f"Loading {RAW} ...")
    df = pd.read_csv(
        RAW,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"],
        dtype={"tweet_id": "Int64", "in_response_to_tweet_id": "Int64", "author_id": "string"},
    )
    print(f"  total rows: {len(df):,}")

    # index every tweet by id for O(1) parent lookup
    by_id = df.set_index("tweet_id", drop=False)

    # brand's own replies: authored by Uber_Support, not inbound
    brand = df[(df["author_id"] == BRAND) & (df["inbound"] == False)].copy()
    print(f"  {BRAND} outbound replies: {len(brand):,}")

    rows = []
    seen_customer = set()
    for _, r in brand.iterrows():
        parent_id = r["in_response_to_tweet_id"]
        if pd.isna(parent_id) or parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):          # duplicate ids -> take first
            parent = parent.iloc[0]
        if parent["inbound"] != True:                 # parent must be a real customer message
            continue
        cust_id = int(parent["tweet_id"])
        if cust_id in seen_customer:                  # keep first brand reply per customer tweet
            continue
        cust_msg = clean(parent["text"])
        brand_reply = clean(r["text"])
        if len(cust_msg) < 5 or len(brand_reply) < 5:
            continue
        seen_customer.add(cust_id)
        rows.append({
            "pair_id": f"uc_{cust_id}",
            "customer_tweet_id": cust_id,
            "customer_msg": cust_msg,
            "brand_reply": brand_reply,
            "created_at": parent["created_at"],
        })

    pairs = pd.DataFrame(rows).drop_duplicates("pair_id").reset_index(drop=True)
    print(f"  reconstructed pairs: {len(pairs):,}")

    # deterministic shuffle, then split grounding corpus vs held-out golden pool (disjoint)
    pairs = pairs.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pool = pairs.iloc[:N_POOL].copy()
    grounding = pairs.iloc[N_POOL:N_POOL + N_GROUNDING].copy()

    grounding.to_csv(PAIRS_OUT, index=False)
    pool.to_csv(POOL_OUT, index=False)
    print(f"\nWrote {len(grounding):,} grounding pairs -> {PAIRS_OUT}")
    print(f"Wrote {len(pool):,} held-out msgs   -> {POOL_OUT}")
    print("These two files are DISJOINT: the golden set is never drawn from the retrieval corpus.")


if __name__ == "__main__":
    main()
