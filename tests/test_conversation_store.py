"""Tests for the shared conversation store + state machine and the raw feedback log."""
import pytest

from src import conversation_store as cs
from src import feedback_log as fl


@pytest.fixture
def store(tmp_path):
    return cs.ConversationStore(root=str(tmp_path / "conversations"))


def test_create_starts_ai_active(store):
    conv = store.create_conversation()
    assert conv["status"] == cs.AI_ACTIVE
    assert conv["messages"] == []
    assert store.get(conv["conversation_id"])["status"] == cs.AI_ACTIVE


def test_add_messages_roles(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    store.add_message(cid, cs.CUSTOMER, "my order hasn't arrived")
    store.add_message(cid, cs.AI, "it should arrive tomorrow")
    got = store.get(cid)
    assert [m["role"] for m in got["messages"]] == [cs.CUSTOMER, cs.AI]
    assert got["messages"][0]["message_id"].startswith("msg_")


def test_invalid_role_rejected(store):
    conv = store.create_conversation()
    with pytest.raises(ValueError):
        store.add_message(conv["conversation_id"], "robot", "hi")


def test_dedup_ignores_identical_consecutive(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    store.add_message(cid, cs.CUSTOMER, "double click", dedup=True)
    dup = store.add_message(cid, cs.CUSTOMER, "double click", dedup=True)
    assert dup is None
    assert len(store.get(cid)["messages"]) == 1


def test_escalation_moves_to_waiting_and_is_idempotent(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    store.escalate(cid, reason="customer marked not helpful", intent="delivery_order")
    got = store.get(cid)
    assert got["status"] == cs.WAITING_FOR_HUMAN
    assert got["escalation_reason"] == "customer marked not helpful"
    assert got["intent"] == "delivery_order"
    assert any(m["role"] == cs.SYSTEM for m in got["messages"])
    # idempotent: escalating again does not add a second system note
    n_sys = sum(1 for m in got["messages"] if m["role"] == cs.SYSTEM)
    store.escalate(cid, reason="again")
    assert sum(1 for m in store.get(cid)["messages"] if m["role"] == cs.SYSTEM) == n_sys


def test_human_reply_moves_waiting_to_active(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    store.escalate(cid, reason="x")
    store.add_message(cid, cs.HUMAN, "Hi, I've checked your order")
    assert store.get(cid)["status"] == cs.HUMAN_ACTIVE


def test_resolution_records_and_resolves(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    store.escalate(cid, reason="x")
    store.add_message(cid, cs.HUMAN, "looking into it")
    store.set_resolution(cid, resolution="Delivery was delayed; rescheduled for tomorrow.",
                         category="delivery", human_agent="alice", resolution_id="abc123")
    got = store.get(cid)
    assert got["status"] == cs.RESOLVED
    assert got["resolution"]["text"].startswith("Delivery was delayed")
    assert got["resolution"]["resolution_id"] == "abc123"


def test_invalid_transition_raises(store):
    conv = store.create_conversation()
    # AI_ACTIVE -> HUMAN_ACTIVE is not allowed (must go through WAITING_FOR_HUMAN)
    with pytest.raises(cs.InvalidTransition):
        store.set_status(conv["conversation_id"], cs.HUMAN_ACTIVE)


def test_resolved_reopens_on_new_customer_message(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    store.escalate(cid, reason="x")
    store.set_resolution(cid, resolution="done")
    assert store.get(cid)["status"] == cs.RESOLVED
    store.add_message(cid, cs.CUSTOMER, "actually it still hasn't arrived")
    got = store.get(cid)
    assert got["status"] == cs.WAITING_FOR_HUMAN   # reopened, not silently dropped
    assert got["messages"][-1]["role"] == cs.CUSTOMER


def test_list_and_counts(store):
    store.create_conversation()
    b = store.create_conversation()
    store.escalate(b["conversation_id"], reason="x")
    assert len(store.list_conversations()) == 2
    assert len(store.list_conversations(status=cs.WAITING_FOR_HUMAN)) == 1
    counts = store.counts_by_status()
    assert counts[cs.AI_ACTIVE] == 1
    assert counts[cs.WAITING_FOR_HUMAN] == 1


def test_feedback_idempotent(store):
    conv = store.create_conversation()
    cid = conv["conversation_id"]
    m = store.add_message(cid, cs.AI, "reply")
    r1 = store.add_feedback(cid, m["message_id"], "helpful")
    r2 = store.add_feedback(cid, m["message_id"], "helpful")
    assert r1["changed"] is True
    assert r2["changed"] is False
    assert store.get(cid)["feedback"][m["message_id"]]["feedback"] == "helpful"


def test_feedback_log_raw_tier(tmp_path):
    path = str(tmp_path / "fb.jsonl")
    fl.log_feedback("conv_1", "msg_1", "helpful", customer_query="q", ai_response="a",
                    intent="billing_payment", path=path)
    fl.log_feedback("conv_1", "msg_2", "not_helpful", path=path)
    recs = fl.load_feedback(path)
    assert len(recs) == 2
    assert all(r["validated"] is False and r["trusted"] is False for r in recs)
    stats = fl.feedback_stats(path)
    assert stats["helpful"] == 1 and stats["not_helpful"] == 1
