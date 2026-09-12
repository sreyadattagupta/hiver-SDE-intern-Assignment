"""Tests for the advanced NLP upgrades: preprocessing, retrieval backends, verifier, needs_context."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline, retrieval
from src.preprocess import clean_for_embedding


def test_preprocess_preserves_crash_distinction():
    a = clean_for_embedding("the app keeps crashing")
    b = clean_for_embedding("my driver crashed the car")
    assert "app" in a and "driver" in b and "car" in b
    assert a != b  # must NOT normalize to the same meaning


def test_preprocess_strips_noise_keeps_words():
    out = clean_for_embedding("@Uber_Support chargedddd twice!! https://t.co/x pls help")
    assert "http" not in out and "@" not in out
    assert "charged" in out and "twice" in out and "please" in out


def test_retrieval_methods_return_contract():
    for method in ("tfidf", "lsa", "hybrid"):
        r = retrieval.retrieve("I was charged twice", k=3, method=method)
        assert len(r) == 3
        for x in r:
            assert {"pair_id", "customer_msg", "brand_reply", "score", "method"} <= set(x)


def test_retrieval_semantic_beats_lexical_on_paraphrase():
    # a paraphrase with few shared words should still retrieve a billing case via semantics
    r = retrieval.retrieve("Uber billed my card two times for one journey", k=3, method="auto")
    joined = " ".join(x["customer_msg"].lower() for x in r)
    assert any(w in joined for w in ["charg", "bill", "twice", "double", "two"])


def test_verifier_flags_invented_url():
    retr = [{"pair_id": "x", "customer_msg": "c", "brand_reply": "we can help", "score": 0.5}]
    v = pipeline.verify_draft("refund pls", "Sure, see https://t.co/FAKE123 now", retr, "billing_payment")
    assert v["ok"] is False and any("invented_url" in i for i in v["issues"])


def test_verifier_flags_unsupported_promise():
    retr = [{"pair_id": "x", "customer_msg": "c", "brand_reply": "we can help", "score": 0.5}]
    v = pipeline.verify_draft("refund", "Good news, we have refunded your trip.", retr, "billing_payment")
    assert v["ok"] is False and any("unsupported_promise" in i for i in v["issues"])


def test_verifier_passes_clean_draft():
    retr = [{"pair_id": "x", "customer_msg": "c", "brand_reply": "sorry, DM us", "score": 0.5}]
    v = pipeline.verify_draft("refund", "So sorry about the double charge — can you share the trip date?",
                              retr, "billing_payment")
    assert v["ok"] is True


def test_needs_context_flag_present_and_calibrated():
    hi = pipeline.classify_intent("I was charged twice, refund me", mode="rules")
    lo = pipeline.classify_intent("hmm", mode="rules")
    assert "needs_context" in hi and "needs_context" in lo
    assert lo["needs_context"] is True  # low-confidence garbage should flag


if __name__ == "__main__":
    import inspect
    n = 0
    for k, fn in sorted(globals().items()):
        if k.startswith("test_") and not inspect.signature(fn).parameters:
            fn(); print("PASS", k); n += 1
    print(f"{n} passed")
