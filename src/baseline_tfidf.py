"""
Baseline 2 (SPEC.md 5): TF-IDF + Logistic Regression classifier (no LLM).

This is the "simple but real" ML baseline the LLM classifier must beat. It has no hand
labels to train on other than the golden set (which is test data), so it trains on WEAK
labels: the transparent rule classifier's predictions over the retrieval corpus
(data/uber_pairs.csv). This is an honest, documented limitation — the baseline can only be
as good as the rules that supervised it, and we report that caveat in the eval.

Evaluation is on the golden set, which is disjoint from the training corpus (no leakage).

Run:  python -m src.baseline_tfidf
"""
from __future__ import annotations

import os
import functools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .pipeline import _classify_rules

PAIRS = os.path.join("data", "uber_pairs.csv")


@functools.lru_cache(maxsize=1)
def _trained_model():
    df = pd.read_csv(PAIRS)
    msgs = df["customer_msg"].fillna("").tolist()
    weak_labels = [_classify_rules(m)["intent"] for m in msgs]
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(msgs, weak_labels)
    return model


def predict(messages):
    return list(_trained_model().predict(list(messages)))


if __name__ == "__main__":
    for m in ["I was charged twice, refund please", "driver cancelled on me", "check dm"]:
        print(m, "->", predict([m])[0])
