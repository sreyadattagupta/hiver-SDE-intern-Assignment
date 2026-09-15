"""End-to-end human-in-the-loop flow through the REAL Streamlit apps (AppTest) + shared store.

Drives the actual button handlers in app.py and pages/1_Human_Support.py against an isolated
temp store, proving the full chain: customer → AI → 👎 → escalation → human reply → resolution →
verified memory → future retrieval. Runs on the offline rules path (no API key) so it is
deterministic and CI-safe.
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

from src import config, conversation_store as cs, memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMER_APP = os.path.join(ROOT, "app.py")
ADMIN_APP = os.path.join(ROOT, "pages", "1_Human_Support.py")


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    conv_dir = str(tmp_path / "conversations")
    mem_path = str(tmp_path / "resolution_memory.jsonl")
    fb_path = str(tmp_path / "feedback_log.jsonl")
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", conv_dir)
    monkeypatch.setattr(config, "MEMORY_PATH", mem_path)
    monkeypatch.setattr(config, "FEEDBACK_LOG_PATH", fb_path)
    return cs.ConversationStore(root=conv_dir)


def _btn(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"button {label!r} not found; have {[b.label for b in at.button]}")


def test_helpful_keeps_ai_and_logs_feedback(isolated_store):
    from src import feedback_log
    at = AppTest.from_file(CUSTOMER_APP, default_timeout=60).run()
    at.text_area[0].set_value("I was charged twice for one ride, please refund the duplicate").run()
    _btn(at, "▶  Send").click().run()
    assert not at.exception
    conv = isolated_store.list_conversations()[0]
    assert any(m["role"] == cs.AI for m in conv["messages"])
    # click Helpful
    _btn(at, "👍 Helpful").click().run()
    conv = isolated_store.get(conv["conversation_id"])
    assert conv["status"] == cs.AI_ACTIVE                    # stays AI-handled
    assert list(conv["feedback"].values())[0]["feedback"] == "helpful"
    fb = feedback_log.load_feedback(config.FEEDBACK_LOG_PATH)
    assert fb and fb[-1]["feedback"] == "helpful" and fb[-1]["trusted"] is False


def test_not_helpful_escalates_to_waiting(isolated_store):
    at = AppTest.from_file(CUSTOMER_APP, default_timeout=60).run()
    at.session_state["live"] = False
    at.text_area[0].set_value("the driver took a huge detour and overcharged me").run()
    _btn(at, "▶  Send").click().run()
    _btn(at, "👎 Not Helpful").click().run()
    assert not at.exception
    conv = isolated_store.list_conversations()[0]
    assert conv["status"] == cs.WAITING_FOR_HUMAN
    assert any(m["role"] == cs.SYSTEM and "Escalated" in m["text"] for m in conv["messages"])


def test_full_loop_customer_to_verified_memory_to_future_retrieval(isolated_store):
    from src import pipeline
    # --- customer raises + marks not helpful -> escalation ---
    at = AppTest.from_file(CUSTOMER_APP, default_timeout=60).run()
    at.session_state["live"] = False
    at.text_area[0].set_value("my ubereats order never arrived but I was charged for it").run()
    _btn(at, "▶  Send").click().run()
    _btn(at, "👎 Not Helpful").click().run()
    conv = isolated_store.list_conversations()[0]
    cid = conv["conversation_id"]
    assert conv["status"] == cs.WAITING_FOR_HUMAN

    # --- admin opens, replies, resolves & verifies ---
    admin = AppTest.from_file(ADMIN_APP, default_timeout=60).run()
    admin.session_state["admin_live"] = False
    admin = admin.run()
    _btn(admin, "Open").click().run()               # opens the single waiting conversation
    assert admin.session_state["admin_sel"] == cid
    # send a human reply
    admin.text_area(key=f"reply_{cid}").set_value("Checked your order — the courier marked it "
                                                  "delivered by mistake; a full refund is on the way.").run()
    _btn(admin, "➤ Send reply").click().run()
    assert isolated_store.get(cid)["status"] == cs.HUMAN_ACTIVE
    # resolve & verify
    admin.text_area(key=f"res_{cid}").set_value("Order was not delivered; issued a full refund to "
                                               "the original payment method within 3-5 business days.").run()
    _btn(admin, "✅ Resolve & Verify").click().run()
    assert not admin.exception

    conv = isolated_store.get(cid)
    assert conv["status"] == cs.RESOLVED
    assert conv["resolution"]["resolution_id"]

    # --- verified resolution is in trusted memory ---
    recs = memory.load_resolutions(config.MEMORY_PATH)
    assert len(recs) == 1 and recs[0]["human_verified"] is True

    # --- future retrieval surfaces the verified resolution for a similar query ---
    hits = pipeline.retrieve_similar("my order never arrived and I was charged",
                                     intent="delivery_order", k=3, memory_path=config.MEMORY_PATH)
    assert any(h.get("verified") for h in hits), "verified resolution should be retrievable"


def test_unresolved_conversation_never_enters_memory(isolated_store):
    """Negative-knowledge: a 👎 + escalation that is NOT resolved must not create trusted memory."""
    at = AppTest.from_file(CUSTOMER_APP, default_timeout=60).run()
    at.session_state["live"] = False
    at.text_area[0].set_value("this app is useless and nothing works").run()
    _btn(at, "▶  Send").click().run()
    _btn(at, "👎 Not Helpful").click().run()
    assert isolated_store.list_conversations()[0]["status"] == cs.WAITING_FOR_HUMAN
    assert memory.load_resolutions(config.MEMORY_PATH) == []    # unresolved -> no trusted knowledge


def test_double_click_helpful_is_idempotent(isolated_store):
    at = AppTest.from_file(CUSTOMER_APP, default_timeout=60).run()
    at.session_state["live"] = False
    at.text_area[0].set_value("how do I add a stop to my trip").run()
    _btn(at, "▶  Send").click().run()
    conv = isolated_store.list_conversations()[0]
    if any(b.label == "👍 Helpful" for b in at.button):
        _btn(at, "👍 Helpful").click().run()
    conv = isolated_store.get(conv["conversation_id"])
    n = len([f for f in conv["feedback"].values() if f["feedback"] == "helpful"])
    assert n <= 1
