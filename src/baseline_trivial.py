"""
Baseline 1 (SPEC.md 5): trivial majority-class classifier.

Predicts the single most common intent in the golden set for every message. This is the
floor any real classifier must clear — if the LLM/rules can't beat "always guess the most
frequent class", they add nothing.

Run:  python -m src.baseline_trivial
"""
from __future__ import annotations

import os
import pandas as pd

GOLDEN = os.path.join("data", "golden_set.csv")


def majority_class(golden_path: str = GOLDEN) -> str:
    return pd.read_csv(golden_path)["true_intent"].value_counts().idxmax()


def predict(messages, golden_path: str = GOLDEN):
    cls = majority_class(golden_path)
    return [cls] * len(messages)


if __name__ == "__main__":
    print("majority class:", majority_class())
