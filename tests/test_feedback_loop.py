"""Tests for the feedback/escalation/verified-memory RAG loop (Steps 2-13, 18 of the spec)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest


# =========================== Task 1: config ===========================

def test_config_constants_present_and_ranged():
    from src import config
    assert 0.0 < config.RESOLUTION_SIM_THRESHOLD < 1.0
    assert 0.0 < config.VERIFIED_TRUST_BOOST < 1.0
    assert 0.0 < config.CONFLICT_SIM_THRESHOLD < 1.0
    assert 0.0 < config.RESOLUTION_TEXT_DIFF_THRESHOLD < 1.0
    assert 0.0 < config.DISSATISFACTION_THRESHOLD < 1.0
    assert 0.0 < config.REPEAT_SIM_THRESHOLD < 1.0
    assert isinstance(config.ESCALATE_SECOND_THUMBS_DOWN, bool)
    assert config.MEMORY_PATH.endswith(".jsonl")


# =========================== Task 2: fitted TF-IDF space ===========================

def test_tfidf_cosine_same_space_scores_relevant_higher():
    from src import retrieval
    q = "I was billed twice for one ride and need a refund"
    texts = [
        "Uber charged my card two times for the same trip, refund please",   # relevant
        "the driver took a long detour and the route was wrong",              # irrelevant
    ]
    import numpy as np
    scores = retrieval.tfidf_cosine(q, texts)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
    assert float(retrieval.tfidf_cosine(q, []).shape[0]) == 0.0


# =========================== Task 3: memory store ===========================

@pytest.fixture()
def tmp_memory(tmp_path):
    return str(tmp_path / "resolution_memory.jsonl")


def _rec(memory, q, r, intent="billing_payment", reason="customer requested human"):
    return memory.build_record(q, r, intent, reason, ai_response="please check payment history")


def test_add_verified_resolution_persists(tmp_memory):
    from src import memory
    rec = _rec(memory, "I was charged twice", "Confirmed duplicate charge; refund issued per policy.")
    out = memory.add_resolution(rec, path=tmp_memory)
    assert out["status"] == "added"
    loaded = memory.load_resolutions(path=tmp_memory)
    assert len(loaded) == 1
    assert loaded[0]["human_verified"] is True
    assert loaded[0]["source"] == "human_resolution"
    assert loaded[0]["resolution_id"] == out["resolution_id"]


def test_duplicate_resolution_prevented(tmp_memory):
    from src import memory
    rec = _rec(memory, "I was charged twice", "Confirmed duplicate charge; refund issued.")
    assert memory.add_resolution(rec, path=tmp_memory)["status"] == "added"
    rec2 = _rec(memory, "I was charged twice", "Confirmed duplicate charge; refund issued.")
    assert memory.add_resolution(rec2, path=tmp_memory)["status"] == "duplicate"
    assert len(memory.load_resolutions(path=tmp_memory)) == 1


def test_unverified_resolution_rejected(tmp_memory):
    from src import memory
    rec = _rec(memory, "app is slow", "we think maybe restart helps")
    rec["human_verified"] = False       # simulate an un-closed / un-verified record
    out = memory.add_resolution(rec, path=tmp_memory)
    assert out["status"] == "rejected_unverified"
    assert memory.load_resolutions(path=tmp_memory) == []


def test_conflict_detected_and_flagged_not_overwritten(tmp_memory):
    from src import memory
    a = _rec(memory, "my promo code will not apply at checkout",
             "Promo codes apply only to your first ride; this one is expired.")
    memory.add_resolution(a, path=tmp_memory)
    b = _rec(memory, "my promo code will not apply at checkout",
             "We manually applied the promo and refunded the difference to your account.")
    out = memory.add_resolution(b, path=tmp_memory)
    assert out["status"] == "added"
    assert out["conflict_flag"] is True
    assert len(memory.load_resolutions(path=tmp_memory)) == 2   # both kept, none overwritten


# =========================== Task 4: resolution retrieval ===========================

def test_verified_resolution_retrieved_when_relevant(tmp_memory):
    from src import memory, resolution_retrieval
    rec = memory.build_record(
        "I was charged twice for the same ride",
        "We confirmed a duplicate charge and the duplicate will be refunded per policy.",
        "billing_payment", "customer rejected AI answer and repeated the issue")
    memory.add_resolution(rec, path=tmp_memory)
    # a realistic re-contact reusing the key content words (see the paraphrase-ceiling note in
    # config.RESOLUTION_SIM_THRESHOLD: zero-overlap paraphrases are a documented lexical limitation)
    hits = resolution_retrieval.retrieve_resolutions(
        "charged twice for one ride, still not refunded", intent="billing_payment", k=3, path=tmp_memory)
    assert len(hits) == 1
    assert hits[0]["source"] == "human_resolution"
    assert hits[0]["verified"] is True
    assert "duplicate" in hits[0]["brand_reply"].lower()


def test_irrelevant_resolution_not_retrieved(tmp_memory):
    from src import memory, resolution_retrieval
    rec = memory.build_record(
        "I was charged twice for the same ride",
        "We confirmed a duplicate charge and refunded it.",
        "billing_payment", "reason")
    memory.add_resolution(rec, path=tmp_memory)
    hits = resolution_retrieval.retrieve_resolutions(
        "how do I change my profile photo", intent="general_query", k=3, path=tmp_memory)
    assert hits == []


def test_intent_mismatch_excludes_resolution(tmp_memory):
    from src import memory, resolution_retrieval
    rec = memory.build_record(
        "I was charged twice for the same ride",
        "We confirmed a duplicate charge and refunded it.",
        "billing_payment", "reason")
    memory.add_resolution(rec, path=tmp_memory)
    # same words, but caller asserts a different intent -> excluded
    hits = resolution_retrieval.retrieve_resolutions(
        "I was charged twice for the same ride", intent="trip_issue", k=3, path=tmp_memory)
    assert hits == []


def test_empty_memory_returns_empty(tmp_memory):
    from src import resolution_retrieval
    assert resolution_retrieval.retrieve_resolutions("anything", path=tmp_memory) == []


# =========================== Task 5: dissatisfaction detection ===========================

def test_explicit_helpful_not_dissatisfied():
    from src import satisfaction
    out = satisfaction.detect_dissatisfaction("thanks that worked", explicit_feedback="helpful", mode="rules")
    assert out["dissatisfied"] is False


def test_explicit_human_request_dissatisfied():
    from src import satisfaction
    out = satisfaction.detect_dissatisfaction("connect me to support", explicit_feedback="human_request", mode="rules")
    assert out["dissatisfied"] is True
    assert "explicit_human_request" in out["signals"]


def test_implicit_rejection_phrases_detected():
    from src import satisfaction
    for m in ["That's not what I asked.", "this isn't helping", "you're not understanding me",
              "that answer is wrong", "I want to talk to a human"]:
        out = satisfaction.detect_dissatisfaction(m, mode="rules")
        assert out["dissatisfied"] is True, m


def test_plain_technical_complaint_not_dissatisfied():
    from src import satisfaction
    # a bare technical description is NOT frustration (spec Step 3 — the crucial distinction)
    out = satisfaction.detect_dissatisfaction("the app keeps crashing when I book", mode="rules")
    assert out["dissatisfied"] is False
    assert out["score"] < 0.6


def test_repeated_question_detected():
    from src import satisfaction
    history = ["the app keeps crashing when I open it"]
    out = satisfaction.detect_dissatisfaction(
        "I already told you the app keeps crashing and your answer did not help",
        history=history, mode="rules")
    assert out["dissatisfied"] is True
    assert any("repeat" in s for s in out["signals"])


# =========================== Task 6: escalation engine ===========================

def test_escalate_on_explicit_human_request():
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.9},
                            "I want to talk to a human",
                            feedback={"explicit": "human_request", "thumbs_down_count": 0})
    assert out["route"] == "HUMAN"
    assert "explicit_human_request" in out["signals"]


def test_first_thumbs_down_does_not_force_escalation():
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.9},
                            "ok",
                            feedback={"explicit": "not_helpful", "thumbs_down_count": 1})
    assert out["route"] == "AI"


def test_second_thumbs_down_escalates():
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.9},
                            "still not helpful",
                            feedback={"explicit": "not_helpful", "thumbs_down_count": 2})
    assert out["route"] == "HUMAN"


def test_safety_incident_escalates():
    from src import escalation
    out = escalation.decide({"intent": "safety_incident", "confidence": 0.9},
                            "my driver assaulted me")
    assert out["route"] == "HUMAN"
    assert any("safety" in s for s in out["signals"])


def test_low_confidence_escalates():
    from src import escalation
    out = escalation.decide({"intent": "general_query", "confidence": 0.30}, "uh what")
    assert out["route"] == "HUMAN"
    assert any("confidence" in s for s in out["signals"])


def test_groundedness_failure_escalates():
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.9}, "refund me",
                            draft_meta={"verification": {"ok": False, "issues": ["invented_url:x"]}})
    assert out["route"] == "HUMAN"
    assert "groundedness_failure" in out["signals"]


def test_llm_failure_escalates_safely():
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.9}, "refund me",
                            draft_meta={"llm_failed": True})
    assert out["route"] == "HUMAN"
    assert "llm_failure" in out["signals"]


def test_happy_path_stays_ai():
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.9},
                            "how do I update my payment method")
    assert out["route"] == "AI"
    assert 0.0 <= out["confidence"] <= 1.0


def test_first_contact_complaint_not_escalated():
    # regression: a strong FIRST-contact complaint ("charged twice, want my money back") is a
    # problem report, NOT dissatisfaction with support — must stay AI. No history / no explicit
    # feedback => implicit-dissatisfaction escalation is not applicable (and no LLM call is made).
    from src import escalation
    out = escalation.decide({"intent": "billing_payment", "confidence": 0.95},
                            "I was charged twice for one ride and I want my money back")
    assert out["route"] == "AI", out["signals"]


# =========================== Task 7: pipeline wiring ===========================

def test_pipeline_backcompat_no_memory_unchanged():
    # with no memory file, retrieve_similar returns the historical contract unchanged
    from src import pipeline
    r = pipeline.retrieve_similar("driver cancelled my ride", k=3, memory_path="does_not_exist.jsonl")
    assert len(r) == 3
    for item in r:
        assert set(["pair_id", "customer_msg", "brand_reply", "score"]).issubset(item)


def test_pipeline_prefers_relevant_verified_resolution(tmp_memory):
    from src import pipeline, memory
    rec = memory.build_record(
        "I was charged twice for the same ride",
        "We confirmed a duplicate charge and it will be refunded per policy.",
        "billing_payment", "customer repeated the billing issue")
    memory.add_resolution(rec, path=tmp_memory)
    r = pipeline.retrieve_similar("charged twice for one ride, still not refunded",
                                  k=3, intent="billing_payment", memory_path=tmp_memory)
    assert r[0]["source"] == "human_resolution"
    assert r[0]["verified"] is True
    # historical items still present in the merged list
    assert any(item.get("source") == "historical" for item in r)


# =========================== Task 8: UI state ===========================

def test_ui_state_thumbs_down_counter_and_escalation_flag():
    from src import ui_state
    turns = [{"role": "ai", "turn_id": 0, "intent": "billing_payment", "feedback": None,
              "escalated": False, "text": "please check payment history"}]
    # first thumbs down -> recorded, not escalated
    s1 = ui_state.apply_feedback(turns, turn_id=0, feedback="not_helpful",
                                 message="that didn't help", history=[])
    assert s1["turns"][0]["feedback"] == "not_helpful"
    assert s1["thumbs_down_count"] == 1
    assert s1["escalated"] is False
    # explicit human request -> escalates immediately
    s2 = ui_state.apply_feedback(s1["turns"], turn_id=0, feedback="human_request",
                                 message="talk to a human", history=[])
    assert s2["escalated"] is True
    assert s2["turns"][0]["escalated"] is True


def test_ui_state_build_escalation_record_has_context():
    from src import ui_state
    rec = ui_state.build_escalation_record(
        conversation_id="c1", original_message="I was charged twice",
        history=["I was charged twice"], intent="billing_payment",
        ai_response="check history", retrieved=[{"pair_id": "uc_1", "score": 0.4}],
        safety_status="no", reason="customer requested human")
    for key in ["conversation_id", "original_message", "history", "intent", "ai_response",
                "retrieved_evidence", "safety_status", "reason", "timestamp", "status"]:
        assert key in rec
    assert rec["status"] == "ESCALATED"


# =========================== Task 9: evaluation ===========================

def test_memory_eval_shows_recall_and_ranking(tmp_memory):
    from src import memory, memory_eval
    memory.add_resolution(memory.build_record(
        "I was charged twice for the same ride",
        "We confirmed a duplicate charge and refunded it per policy.",
        "billing_payment", "repeated billing issue"), path=tmp_memory)
    res = memory_eval.run(memory_path=tmp_memory)
    assert res["recall_at_k"] == 1.0
    assert res["verified_ranked_first"] is True
    assert res["below_threshold_excluded"] is True
    assert res["intent_mismatch_excluded"] is True
