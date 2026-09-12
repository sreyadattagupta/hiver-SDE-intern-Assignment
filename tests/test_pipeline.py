"""Contract + behavior tests for the agent. Run: python -m pytest tests/ -q  (or python tests/test_pipeline.py)"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pipeline


def test_classify_contract():
    out = pipeline.classify_intent("I was charged twice, refund please", mode="rules")
    assert set(["intent", "confidence", "runner_up", "method"]).issubset(out)
    assert out["intent"] in pipeline.INTENTS
    assert 0.0 <= out["confidence"] <= 1.0


def test_retrieve_contract():
    r = pipeline.retrieve_similar("driver cancelled my ride", k=3)
    assert len(r) == 3
    for item in r:
        assert set(["pair_id", "customer_msg", "brand_reply", "score"]).issubset(item)
        assert -1.01 <= item["score"] <= 1.01  # cosine/fused; reranker may reorder (not strictly desc)


def test_draft_contract():
    r = pipeline.retrieve_similar("I need a refund", k=2)
    d = pipeline._draft_rules("I need a refund", r)
    assert set(["draft", "cited_example_id", "justification", "method"]).issubset(d)
    assert d["cited_example_id"] == r[0]["pair_id"]


def test_route_safety_escalates():
    c = {"intent": "safety_incident", "confidence": 0.9, "runner_up": "trip_issue"}
    rd = pipeline.route_decision(c, "my driver assaulted me", [])
    assert rd["decision"] == "HUMAN"
    assert "safety" in rd["reason"].lower()


def test_route_ai_happy_path():
    c = {"intent": "billing_payment", "confidence": 0.9, "runner_up": "trip_issue"}
    rd = pipeline.route_decision(c, "how do I update my payment method", [])
    assert rd["decision"] == "AI"


def test_safety_context_app_vs_car_crash():
    # context-aware NLP: software 'crash' is NOT a safety incident, a vehicle 'crash' IS
    assert pipeline.detect_safety("the app keeps crashing every time I book", mode="rules")["is_safety"] is False
    assert pipeline.detect_safety("our car crashed into a pole on the highway", mode="rules")["is_safety"] is True


def test_safety_negation_handled():
    assert pipeline.detect_safety("no accident, just a late pickup", mode="rules")["is_safety"] is False


def test_naive_bug_fixed_by_context():
    # the OLD blind matcher still flags 'crash' (the documented bug); the NEW context detector does not
    assert "crash" in pipeline.naive_safety_hits("the app keeps crashing")          # legacy bug
    assert pipeline.detect_safety("the app keeps crashing", mode="rules")["is_safety"] is False  # fixed


def test_safety_incident_still_escalates():
    c = {"intent": "safety_incident", "confidence": 0.9, "runner_up": "trip_issue"}
    rd = pipeline.route_decision(c, "the driver assaulted me", [])
    assert rd["decision"] == "HUMAN"


def test_repeat_contact_escalates():
    c = {"intent": "billing_payment", "confidence": 0.9, "runner_up": "trip_issue"}
    rd = pipeline.route_decision(c, "still waiting", ["m1", "m2", "m3"])
    assert rd["decision"] == "HUMAN"
    assert "repeat" in rd["reason"].lower()


def test_edge_inputs_do_not_crash():
    for m in ["", "   ", "😡🔥", "zzz qqq", "refund " * 3000]:
        c = pipeline.classify_intent(m, mode="rules")
        assert c["intent"] in pipeline.INTENTS
        r = pipeline.retrieve_similar(m, k=3)
        assert isinstance(r, list)
        assert pipeline.route_decision(c, m, [])["decision"] in ("AI", "HUMAN")


def test_llm_failure_falls_back_and_is_reported(monkeypatch):
    # simulate the LLM path raising -> auto mode must degrade to rules AND say so in "method"
    def boom(message):
        raise RuntimeError("simulated API outage")
    monkeypatch.setattr(pipeline, "_classify_llm", boom)
    monkeypatch.setattr(pipeline.llm, "llm_available", lambda: True)
    out = pipeline.classify_intent("I was charged twice", mode="auto")
    assert out["method"].startswith("rules")
    assert "llm_error" in out["method"]  # honest: failure is reported, not silently faked


if __name__ == "__main__":
    # bare runner (no pytest): run zero-arg tests; fixture-based ones need `pytest`
    import inspect
    ran = 0
    for k, fn in sorted(globals().items()):
        if not k.startswith("test_"):
            continue
        if inspect.signature(fn).parameters:  # needs a fixture (e.g. monkeypatch)
            print("SKIP (run via pytest):", k)
            continue
        fn()
        print("PASS", k)
        ran += 1
    print(f"\n{ran} tests passed (run `python -m pytest tests/` for the full suite)")
