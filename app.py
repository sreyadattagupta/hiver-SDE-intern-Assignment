"""Uber_Support AI — Customer Support console (human-in-the-loop).

Thin viewer over the real agent (src/pipeline.py) + the shared conversation store
(src/conversation_store.py). Every value shown — intent, confidence, retrieval evidence, draft,
safety, routing — is a real return value of a real pipeline call, filled live as each stage runs.
Conversations, feedback, escalations and human replies are persisted to disk so the separate
Human Support dashboard (see the sidebar) shares the same state.

Flow:  Customer → AI (4-stage pipeline) → 👍/👎/💬 → escalate → Human Support → resolution → memory.

Run:  streamlit run app.py     (the Human Support page appears automatically in the sidebar)
"""
import streamlit as st

from src import pipeline, llm, escalation, feedback_log
from src import conversation_store as cs
from src import ui_components as ui
from src.ui_components import esc, md

st.set_page_config(page_title="Uber Support AI — Customer", page_icon="🚕", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🚕 Uber Support AI")
    st.caption("Use the page nav above to switch between the **Customer Console** and the "
               "**Human Support** dashboard. Both share one backend — escalations raised here "
               "appear on the dashboard.")

store = cs.ConversationStore()

# ----------------------------- session state -----------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = store.create_conversation()["conversation_id"]
if "msg" not in st.session_state:
    st.session_state.msg = ""
if "last_run" not in st.session_state:
    st.session_state.last_run = None          # 4 stage dicts from the most recent real pipeline run
if "live" not in st.session_state:
    st.session_state.live = True

# Clear the input box on the run AFTER a send. Streamlit forbids writing a widget-keyed state value
# once the widget is instantiated, so we do it here — before the text_area is created this run.
if st.session_state.get("_clear_msg"):
    st.session_state.msg = ""
    st.session_state._clear_msg = False

CONV_ID = st.session_state.conversation_id


def _conv():
    c = store.get(CONV_ID)
    if c is None:                              # store wiped underneath us -> start fresh
        c = store.create_conversation(conversation_id=CONV_ID)
    return c


# ----------------------------- top bar -----------------------------
active = (f"LLM chain · {' → '.join(llm.provider_chain())}"
          if llm.llm_available() else "rules (offline)")
ui.render_top_bar(
    "🚕 Uber Support AI",
    f'<span class="dot on"></span><span class="online">Agent Online</span><br/>'
    f'<b>Model:</b> {esc(active)} &nbsp;·&nbsp; <b>Human Support:</b> open the sidebar ▶')

TITLES = ["Intent Classification", "Semantic Retrieval", "Response Generation", "Safety & Routing"]


def stage_card(n, title, status, kv_html=""):
    icon = {"pending": "○", "running": "●", "done": "✓"}[status]
    stt = {"pending": "pend", "running": "run", "done": "done"}[status]
    label = {"pending": "idle", "running": "running…", "done": "completed"}[status]
    cls = {"pending": "", "running": "run", "done": "done"}[status]
    return (f'<div class="pstage {cls}"><div class="top">{icon} {n}. {esc(title)}'
            f'<span class="st {stt}">{label}</span></div>{kv_html}</div>')


def bar(pct, color):
    return f'<div class="mbar"><span style="width:{max(4,int(pct*100))}%;background:{color};"></span></div>'


def why(text):
    return f'<div class="pwhy"><span class="lb">Why:</span> {esc(text)}</div>'


# ----------------------------- layout -----------------------------
left, right = st.columns([1, 2.7], gap="large")

with left:
    st.markdown('<div class="panel-h">⚙ Agent Run</div>', unsafe_allow_html=True)
    ph = []
    left_scroll = st.container(height=470, border=False)
    with left_scroll:
        for i, t in enumerate(TITLES):
            slot = st.empty()
            ph.append(slot)
            if i < 3:
                st.markdown('<div class="parrow">↓</div>', unsafe_allow_html=True)


def paint_left(stages):
    """Render the left panel from a list of 4 stage dicts (or all-idle if None). Reflects the REAL
    most-recent pipeline run — never a fake tick."""
    for i, title in enumerate(TITLES):
        s = (stages or [None, None, None, None])[i]
        if s is None:
            ph[i].markdown(stage_card(i + 1, title, "pending"), unsafe_allow_html=True)
        else:
            ph[i].markdown(stage_card(i + 1, title, s["status"], s["kv"]), unsafe_allow_html=True)


def set_stage_live(i, status, kv=""):
    ph[i].markdown(stage_card(i + 1, TITLES[i], status, kv), unsafe_allow_html=True)


with right:
    md('<div class="chat-h"><span class="t">Customer Support</span>'
       '<span class="s">Live customer ↔ AI ↔ human conversation</span></div>')
    chat_area = st.container(height=360, border=False)

    st.markdown('<div class="mini">Quick examples</div>', unsafe_allow_html=True)
    EXAMPLES = {
        "💳 Billing": "I was charged twice for one ride and no one is responding. Refund me!",
        "🚨 Safety": "our uber driver was drunk and crashed the car, I felt unsafe",
        "📱 App crash": "the app keeps crashing every time I try to book a ride",
        "🍔 Delivery": "my ubereats order never arrived but I was charged for it",
    }
    ecols = st.columns(len(EXAMPLES))
    for col, (lab, txt) in zip(ecols, EXAMPLES.items()):
        if col.button(lab, use_container_width=True):
            st.session_state.msg = txt
            st.rerun()

    msg = st.text_area("Customer message", key="msg", height=80,
                       placeholder="Type a customer message…", label_visibility="collapsed")
    c_send, c_new, c_live = st.columns([3, 1.3, 1.4])
    send = c_send.button("▶  Send", type="primary", use_container_width=True)
    if c_new.button("New chat", use_container_width=True):
        st.session_state.conversation_id = store.create_conversation()["conversation_id"]
        st.session_state.last_run = None
        st.rerun()
    st.session_state.live = c_live.toggle("Live", value=st.session_state.live,
                                          help="Auto-refresh to receive human-agent replies")


# ----------------------------- feedback handling -----------------------------
def _thumbs_down_count(conv):
    return sum(1 for f in conv["feedback"].values() if f.get("feedback") == "not_helpful")


def handle_feedback(message_id, feedback, ai_text, intent):
    conv = _conv()
    customer_msgs = [m["text"] for m in conv["messages"] if m["role"] == cs.CUSTOMER]
    query = customer_msgs[-1] if customer_msgs else ""
    store.add_feedback(CONV_ID, message_id, feedback)
    feedback_log.log_feedback(CONV_ID, message_id, feedback, customer_query=query,
                              ai_response=ai_text, intent=intent)
    # Customer-facing policy (human-in-the-loop product flow): an explicit 👎 Not Helpful or a
    # 💬 human request escalates immediately — the customer is explicitly asking for better help.
    # (The escalation ENGINE still supports its 2nd-👎 policy for autonomous routing; this is an
    # explicit-signal override.)
    if feedback in ("not_helpful", "human_request"):
        reason = ("the customer explicitly requested a human agent" if feedback == "human_request"
                  else "the customer marked the AI response Not Helpful")
        store.escalate(CONV_ID, reason=reason, intent=intent)
    st.rerun()


# ----------------------------- run the real pipeline -----------------------------
def run_pipeline(message):
    """Execute the 4 real stages, updating the left panel live, and return (ai_text, tag, stages,
    decision, intent, retrieved, safety)."""
    stages = [None, None, None, None]

    set_stage_live(0, "running")
    c = pipeline.classify_intent(message)
    color = ui.INTENT_COLORS.get(c["intent"], "#64748b")
    nc = ' · <span class="mini">needs_context</span>' if c.get("needs_context") else ""
    kv0 = (f'<div class="kv">model: <b>{esc(c["method"])}</b>{nc}<br/>'
           f'intent: <span class="pbadge" style="background:{color};">{esc(c["intent"])}</span> '
           f'<span class="mini">/ runner-up {esc(c["runner_up"])}</span><br/>'
           f'confidence: <b>{c["confidence"]:.2f}</b>{bar(c["confidence"], color)}</div>'
           + why(c.get("reason", f"classified as {c['intent']}")))
    stages[0] = {"status": "done", "kv": kv0}
    set_stage_live(0, "done", kv0)

    set_stage_live(1, "running")
    retrieved = pipeline.retrieve_similar(message, k=3, intent=c["intent"])
    top = retrieved[0] if retrieved else {"pair_id": "-", "score": 0, "customer_msg": "", "method": "-"}
    n_verified = sum(1 for r in retrieved if r.get("verified"))
    ev = "".join(f'<div class="evrow">{"🧠 " if r.get("verified") else ""}[{esc(r["pair_id"])}] '
                 f'{r["score"]:.3f} · {esc(r["customer_msg"][:52])}</div>' for r in retrieved)
    if top.get("verified"):
        rwhy = (f'top match is a HUMAN-VERIFIED resolution (sim {top["score"]:.3f}) — matched the '
                f'intent and cleared the similarity threshold, so it grounds the reply with higher trust.')
    else:
        rwhy = (f'closest of 1,500 real historical cases by {top["method"]} similarity (sim '
                f'{top["score"]:.3f}) grounds the reply.')
    kv1 = (f'<div class="kv">retriever: <b>{esc(top["method"])}</b> · top sim <b>{top["score"]:.3f}</b>'
           + (f' · <span class="mini">{n_verified} verified</span>' if n_verified else "")
           + f'{ev}</div>' + why(rwhy))
    stages[1] = {"status": "done", "kv": kv1}
    set_stage_live(1, "done", kv1)

    set_stage_live(2, "running")
    d = pipeline.draft_reply(message, retrieved, intent=c["intent"])
    v = d.get("verification", {"ok": True, "issues": []})
    vtxt = "passed" if v["ok"] else "FAILED (" + ", ".join(v["issues"]) + ")"
    kv2 = (f'<div class="kv">model: <b>{esc(d["method"])}</b><br/>'
           f'cited: <b>{esc(d["cited_example_id"])}</b> · verifier: '
           f'<b style="color:{"#16a34a" if v["ok"] else "#dc2626"};">{esc(vtxt)}</b></div>'
           + why(d.get("justification") or "grounded in the cited historical reply"))
    stages[2] = {"status": "done", "kv": kv2}
    set_stage_live(2, "done", kv2)

    set_stage_live(3, "running")
    safety = pipeline.detect_safety(message)
    prior = [m["text"] for m in _conv()["messages"] if m["role"] == cs.CUSTOMER]
    decision = escalation.decide(
        c, message, history=prior[:-1] if prior else [],
        feedback={"explicit": None, "thumbs_down_count": 0},
        draft_meta={"verification": v, "retrieval_top_score": top["score"],
                    "llm_failed": d["method"].startswith("rules (llm_error")})
    dcolor = "#dc2626" if decision["route"] == "HUMAN" else "#16a34a"
    sig = (" · signals: " + ", ".join(decision["signals"])) if decision["signals"] else ""
    kv3 = (f'<div class="kv">safety: <b>{"YES" if safety["is_safety"] else "no"}</b> '
           f'<span class="mini">({esc(safety["method"])})</span> — {esc(safety.get("reason",""))}<br/>'
           f'decision: <span class="pbadge" style="background:{dcolor};">{esc(decision["route"])}</span>'
           f'<span class="mini">{esc(sig)}</span></div>' + why(decision["reason"]))
    stages[3] = {"status": "done", "kv": kv3}
    set_stage_live(3, "done", kv3)

    tag = f'{decision["route"]} · {c["intent"]} · {d["method"]}'
    return {"ai_text": d["draft"] or "(no draft)", "tag": tag, "stages": stages,
            "decision": decision, "intent": c["intent"], "confidence": c["confidence"],
            "retrieved": retrieved, "safety": safety}


# ----------------------------- handle Send -----------------------------
if send and msg.strip():
    conv = _conv()
    added = store.add_message(CONV_ID, cs.CUSTOMER, msg.strip(), dedup=True)
    conv = _conv()
    if conv["status"] in (cs.WAITING_FOR_HUMAN, cs.HUMAN_ACTIVE):
        # a human owns this conversation now — the customer's message is queued for the agent,
        # the AI does not answer over the top of a human.
        st.session_state._clear_msg = True
        st.rerun()
    else:
        out = run_pipeline(msg.strip())
        ev = [{"pair_id": r.get("pair_id"), "score": r.get("score")} for r in out["retrieved"]]
        ai_msg = store.add_message(CONV_ID, cs.AI, out["ai_text"],
                                   meta={"tag": out["tag"], "intent": out["intent"],
                                         "confidence": out["confidence"]})
        # persist the stage summary + intent so the AI turn can be graded / escalated later
        c2 = store.get(CONV_ID)
        c2["intent"] = out["intent"]
        c2["retrieved_evidence"] = ev
        store._write(c2)
        st.session_state.last_run = out["stages"]
        if out["decision"]["route"] == "HUMAN":
            store.escalate(CONV_ID, reason=out["decision"]["reason"], intent=out["intent"],
                           safety_status="YES" if out["safety"]["is_safety"] else "no",
                           retrieved_evidence=ev)
        st.session_state._clear_msg = True
        st.rerun()

# paint left panel from the last real run (idle before the first run)
paint_left(st.session_state.last_run)


# ----------------------------- render chat (+ feedback / waiting / resolved) -----------------------------
def render_chat_body():
    conv = _conv()
    status = conv["status"]
    if not conv["messages"]:
        md('<div class="emptead">No conversation yet — type a customer message and click '
           '<b>Send</b>.</div>')
        return
    last_ai_id = None
    for m in conv["messages"]:
        ui.render_message(m)
        if m["role"] == cs.AI:
            last_ai_id = m["message_id"]

    # feedback controls: only on the latest AI turn while the AI still owns the conversation
    if status == cs.AI_ACTIVE and last_ai_id is not None:
        ai_msg = next(m for m in conv["messages"] if m["message_id"] == last_ai_id)
        fb = conv["feedback"].get(last_ai_id, {}).get("feedback")
        if fb is None:
            b1, b2, b3 = st.columns(3)
            if b1.button("👍 Helpful", key=f"up_{last_ai_id}", use_container_width=True):
                handle_feedback(last_ai_id, "helpful", ai_msg["text"], conv.get("intent", ""))
            if b2.button("👎 Not Helpful", key=f"down_{last_ai_id}", use_container_width=True):
                handle_feedback(last_ai_id, "not_helpful", ai_msg["text"], conv.get("intent", ""))
            if b3.button("💬 Talk to Human", key=f"human_{last_ai_id}", use_container_width=True):
                handle_feedback(last_ai_id, "human_request", ai_msg["text"], conv.get("intent", ""))
        elif fb == "helpful":
            md('<div class="fbnote" style="color:#15803d;">✓ You marked this <b>Helpful</b>. '
               'Feedback recorded — AI is handling your request.</div>')

    if status in (cs.WAITING_FOR_HUMAN, cs.HUMAN_ACTIVE):
        if status == cs.WAITING_FOR_HUMAN:
            md('<div class="wait-banner">🟠 <b>Connecting you to a support agent…</b><br/>'
               'A support agent will be with you shortly. You can keep typing to add details.</div>')
        else:
            md('<div class="wait-banner">👤 <b>A support agent is now helping you.</b></div>')

    if status == cs.RESOLVED:
        res = conv.get("resolution") or {}
        md('<div class="resolved-banner">✅ <b>Resolved by Human Support.</b><br/>'
           + (f'{esc(res.get("text",""))}' if res.get("text") else "") + '</div>')
        md('<div class="loopviz">'
           '<span class="n">AI RESPONSE</span>→<span class="n">👎 FEEDBACK</span>→'
           '<span class="n">🔴 ESCALATION</span>→<span class="n">👤 RESOLUTION</span>→'
           '<span class="n ok">✓ VERIFIED</span>→<span class="n ok">🧠 MEMORY</span>→'
           '<span class="n ok">🔎 FUTURE RETRIEVAL</span></div>')


_conv_now = _conv()
_polling = st.session_state.live and _conv_now["status"] in (cs.WAITING_FOR_HUMAN, cs.HUMAN_ACTIVE)

if _polling:
    # customer is waiting on a human — auto-refresh the read-only timeline so human replies appear
    # without a manual refresh. No interactive widgets run inside the timed fragment.
    @st.fragment(run_every=3)
    def _poll():
        with chat_area:
            render_chat_body()
    _poll()
else:
    with chat_area:
        render_chat_body()
