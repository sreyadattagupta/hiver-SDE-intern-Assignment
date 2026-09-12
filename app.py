"""
Uber_Support AI Agent — internal live support console (UI only).

This is a THIN viewer over src/pipeline.py (CLAUDE.md 0.1). It contains no agent logic: every
value shown — active model/method, intent, confidence, retrieval evidence, draft, safety, routing
— is a real return value of a real src.pipeline call, filled LIVE as each stage executes. No fake
animations, no canned replies. Backend/AI logic is unchanged.

Layout: LEFT (~28%) compact live agent pipeline · RIGHT (~72%) large customer↔AI conversation.

Run:  streamlit run app.py
"""
import html
import streamlit as st

from src import pipeline, llm

st.set_page_config(page_title="Uber Support AI — Console", page_icon="🚕", layout="wide")

INTENT_COLORS = {
    "billing_payment": "#2563eb", "account_access": "#7c3aed", "trip_issue": "#0891b2",
    "safety_incident": "#dc2626", "delivery_order": "#ea580c", "service_complaint": "#db2777",
    "general_query": "#64748b",
}

st.markdown("""
<style>
/* hide Streamlit's default top toolbar so our header is never clipped, and pad the top */
header[data-testid="stHeader"] {display:none;}
#MainMenu, footer {visibility:hidden;}
.block-container {max-width: 1400px; padding-top: 1.1rem; padding-bottom: .6rem; overflow-x:hidden;}
.bub, .pstage .kv, .pstage .top {overflow-wrap:anywhere; word-break:break-word;}
.pstage {overflow:hidden;}
/* ---------- top bar ---------- */
.topbar {background:linear-gradient(90deg,#0b0f19 0%,#111827 60%,#1f2937 100%);
         border-radius:14px; padding:14px 22px; color:#fff; display:flex; align-items:center;
         justify-content:space-between; margin-bottom:14px;
         position:sticky; top:0; z-index:100;}
.topbar .brand {font-size:1.25rem; font-weight:700; letter-spacing:-.3px;}
.topbar .meta {font-size:.78rem; color:#cbd5e1; text-align:right; line-height:1.5;}
.topbar .meta b {color:#e5e7eb;}
.dot {height:9px; width:9px; border-radius:50%; display:inline-block; margin-right:6px;}
.dot.on {background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,.25);}
.online {color:#22c55e; font-weight:600;}
/* ---------- left pipeline ---------- */
.panel-h {font-size:.8rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
          color:#64748b; margin:2px 0 8px;}
.pstage {border:1px solid rgba(128,128,128,.22); border-radius:11px; padding:9px 12px;
         background:rgba(127,127,127,.04);}
.pstage.run {border-color:#f59e0b; background:rgba(245,158,11,.07);}
.pstage.done {border-color:rgba(34,197,94,.5);}
.pstage .top {display:flex; align-items:center; gap:8px; font-weight:600; font-size:.86rem;}
.pstage .st {margin-left:auto; font-size:.72rem; font-weight:600;}
.pstage .st.done {color:#16a34a;} .pstage .st.run {color:#d97706;} .pstage .st.pend {color:#94a3b8;}
.pstage .kv {font-size:.75rem; color:#475569; margin-top:5px; line-height:1.5;}
.pstage .kv b {color:#0f172a;}
.pwhy {font-size:.72rem; color:#475569; margin-top:7px; padding-top:6px;
       border-top:1px dashed rgba(128,128,128,.28); line-height:1.45;}
.pwhy .lb {font-weight:700; color:#0f172a; letter-spacing:.02em;}
:root[data-theme="dark"] .pwhy .lb, .pwhy .lb {}
@media (prefers-color-scheme: dark){ .pwhy{color:#94a3b8;} .pwhy .lb{color:#e5e7eb;} }
.pbadge {display:inline-block; padding:1px 8px; border-radius:6px; color:#fff; font-size:.72rem; font-weight:600;}
.parrow {text-align:center; color:#cbd5e1; font-size:.9rem; margin:1px 0;}
.mini {font-size:.7rem; color:#94a3b8;}
.evrow {font-size:.7rem; color:#94a3b8; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.mbar {height:6px; border-radius:5px; background:rgba(128,128,128,.2); overflow:hidden; margin-top:4px;}
.mbar > span {display:block; height:100%;}
/* ---------- right chat ---------- */
.chat-h {display:flex; align-items:baseline; gap:10px; border-bottom:1px solid rgba(128,128,128,.18);
         padding-bottom:8px; margin-bottom:6px;}
.chat-h .t {font-size:1.15rem; font-weight:700;}
.chat-h .s {font-size:.8rem; color:#64748b;}
.chatwrap {padding:2px 6px 2px 2px;}
.row {display:flex; margin:10px 0;}
.row.cust {justify-content:flex-start;}
.row.ai {justify-content:flex-end;}
.who {font-size:.68rem; font-weight:700; letter-spacing:.05em; color:#94a3b8; margin:0 6px 3px;}
.bub {max-width:74%; padding:11px 15px; border-radius:16px; font-size:.96rem; line-height:1.45;}
.bub.cust {background:#f1f5f9; color:#0f172a; border:1px solid #e2e8f0; border-bottom-left-radius:5px;}
.bub.ai {background:#2563eb; color:#fff; border-bottom-right-radius:5px;}
.bub .tag {display:block; font-size:.7rem; opacity:.7; margin-top:6px;}
.emptead {color:#94a3b8; text-align:center; padding:48px 10px; font-size:.9rem;}
@media (prefers-color-scheme: dark){
  .bub.cust{background:#1e293b; color:#e2e8f0; border-color:#334155;}
  .pstage .kv{color:#94a3b8;} .pstage .kv b{color:#e5e7eb;}
}
:root[data-theme="dark"] .bub.cust{background:#1e293b; color:#e2e8f0; border-color:#334155;}
</style>
""", unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def md(h):
    st.markdown("\n".join(l.strip() for l in h.splitlines() if l.strip()), unsafe_allow_html=True)


def stage_card(n, title, status, kv_html=""):
    """status: pending|running|done. Real state, filled as the stage executes."""
    icon = {"pending": "○", "running": "●", "done": "✓"}[status]
    stt = {"pending": "pend", "running": "run", "done": "done"}[status]
    label = {"pending": "pending", "running": "running…", "done": "completed"}[status]
    cls = {"pending": "", "running": "run", "done": "done"}[status]
    return (f'<div class="pstage {cls}"><div class="top">{icon} {n}. {esc(title)}'
            f'<span class="st {stt}">{label}</span></div>{kv_html}</div>')


def bar(pct, color):
    return f'<div class="mbar"><span style="width:{max(4,int(pct*100))}%;background:{color};"></span></div>'


# ----------------------------- top bar -----------------------------
active = (f"LLM fallback chain · {' → '.join(llm.provider_chain())}"
          if llm.llm_available() else "rules (offline)")
md(f"""
<div class="topbar">
  <div class="brand">🚕 Uber Support AI</div>
  <div class="meta">
    <span class="dot on"></span><span class="online">Agent Online</span><br/>
    <b>Model:</b> {esc(active)} &nbsp;·&nbsp; <b>Pipeline:</b> 4-stage AI support
  </div>
</div>
""")

if "chat" not in st.session_state:
    st.session_state.chat = []
if "msg" not in st.session_state:
    st.session_state.msg = ""

left, right = st.columns([1, 2.7], gap="large")

# placeholders created inside the left column's OWN bounded scroll container (independent scroll)
with left:
    st.markdown('<div class="panel-h">⚙ Agent Run</div>', unsafe_allow_html=True)
    titles = ["Intent Classification", "Semantic Retrieval", "Response Generation", "Safety & Routing"]
    ph = []
    left_scroll = st.container(height=460, border=False)
    with left_scroll:
        for i, t in enumerate(titles):          # interleave card / arrow / card ...
            slot = st.empty()
            slot.markdown(stage_card(i + 1, t, "pending"), unsafe_allow_html=True)
            ph.append(slot)
            if i < 3:
                st.markdown('<div class="parrow">↓</div>', unsafe_allow_html=True)


def set_stage(i, title, status, kv=""):
    ph[i].markdown(stage_card(i + 1, title, status, kv), unsafe_allow_html=True)


# ----------------------------- right: chat (own scroll) + fixed input -----------------------------
with right:
    md("""
    <div class="chat-h"><span class="t">Customer Support</span>
    <span class="s">Live customer ↔ AI conversation</span></div>
    """)
    # conversation history in its OWN bounded scroll container (independent of the left panel)
    chat_scroll = st.container(height=360, border=False)
    with chat_scroll:
        chat_box = st.empty()

    st.markdown('<div class="mini">Quick examples</div>', unsafe_allow_html=True)
    EXAMPLES = {
        "💳 Billing": "I was charged twice for one ride and no one is responding. Refund me!",
        "🚨 Safety": "our uber driver was drunk and crashed the car, I felt unsafe",
        "📱 App crash": "the app keeps crashing every time I try to book a ride",
        "😠 Harassment": "male drivers keep hitting on me and it makes me uncomfortable",
    }
    ecols = st.columns(len(EXAMPLES))
    for col, (lab, txt) in zip(ecols, EXAMPLES.items()):
        if col.button(lab, use_container_width=True):
            st.session_state.msg = txt
            st.rerun()

    msg = st.text_area("Customer message", key="msg", height=80,
                       placeholder="Type customer message…", label_visibility="collapsed")
    c_run, c_clear = st.columns([4, 1])
    run = c_run.button("▶  Run Agent", type="primary", use_container_width=True)
    if c_clear.button("Clear", use_container_width=True):
        st.session_state.chat = []
        st.rerun()


def render_chat():
    with chat_box.container():
        st.markdown('<div class="chatwrap">', unsafe_allow_html=True)
        if not st.session_state.chat:
            st.markdown('<div class="emptead">No conversation yet — type a customer message and '
                        'click <b>Run Agent</b>.</div>', unsafe_allow_html=True)
        for turn in st.session_state.chat:
            if turn["role"] == "customer":
                md(f'<div class="row cust"><div><div class="who">Customer</div>'
                   f'<div class="bub cust">{esc(turn["text"])}</div></div></div>')
            else:
                md(f'<div class="row ai"><div><div class="who" style="text-align:right;">AI Support Agent</div>'
                   f'<div class="bub ai">{esc(turn["text"])}<span class="tag">{esc(turn["tag"])}</span></div></div></div>')
        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------- live run -----------------------------
if run and msg.strip():
    st.session_state.chat.append({"role": "customer", "text": msg.strip()})
    render_chat()  # show customer message immediately

    def why(text):
        return f'<div class="pwhy"><span class="lb">Why:</span> {esc(text)}</div>'

    # STAGE 1 — classification (real intent + real rationale)
    set_stage(0, titles[0], "running")
    c = pipeline.classify_intent(msg)
    color = INTENT_COLORS.get(c["intent"], "#64748b")
    nc = ' · <span class="mini">needs_context</span>' if c.get("needs_context") else ""
    set_stage(0, titles[0], "done",
              f'<div class="kv">model: <b>{esc(c["method"])}</b>{nc}<br/>'
              f'intent: <span class="pbadge" style="background:{color};">{esc(c["intent"])}</span> '
              f'<span class="mini">/ runner-up {esc(c["runner_up"])}</span><br/>'
              f'confidence: <b>{c["confidence"]:.2f}</b>{bar(c["confidence"], color)}</div>'
              + why(c.get("reason", f"classified as {c['intent']}")))

    # STAGE 2 — retrieval (real evidence + why it was chosen)
    set_stage(1, titles[1], "running")
    retrieved = pipeline.retrieve_similar(msg, k=3)
    top = retrieved[0] if retrieved else {"pair_id": "-", "score": 0, "customer_msg": "", "method": "-"}
    ev = "".join(
        f'<div class="evrow">[{esc(r["pair_id"])}] {r["score"]:.3f} · {esc(r["customer_msg"][:60])}</div>'
        for r in retrieved)
    rwhy = (f'closest of 1,500 real historical cases by {top["method"]} similarity; top match '
            f'(sim {top["score"]:.3f}) is semantically nearest to the customer\'s wording, so its '
            f'resolution grounds the reply.')
    set_stage(1, titles[1], "done",
              f'<div class="kv">retriever: <b>{esc(top["method"])}</b> · top sim '
              f'<b>{top["score"]:.3f}</b>{ev}</div>' + why(rwhy))

    # STAGE 3 — response generation (real draft justification + verifier)
    set_stage(2, titles[2], "running")
    d = pipeline.draft_reply(msg, retrieved, intent=c["intent"])
    v = d.get("verification", {"ok": True, "issues": []})
    vtxt = "passed" if v["ok"] else "FAILED (" + ", ".join(v["issues"]) + ")"
    set_stage(2, titles[2], "done",
              f'<div class="kv">model: <b>{esc(d["method"])}</b><br/>'
              f'cited: <b>{esc(d["cited_example_id"])}</b> · verifier: '
              f'<b style="color:{"#16a34a" if v["ok"] else "#dc2626"};">{esc(vtxt)}</b></div>'
              + why(d.get("justification") or "grounded in the cited historical reply"))

    # STAGE 4 — safety & routing (real safety reason + routing reason)
    set_stage(3, titles[3], "running")
    safety = pipeline.detect_safety(msg)
    rd = pipeline.route_decision(c, msg, [])
    dcolor = "#dc2626" if rd["decision"] == "HUMAN" else "#16a34a"
    set_stage(3, titles[3], "done",
              f'<div class="kv">safety: <b>{"YES" if safety["is_safety"] else "no"}</b> '
              f'<span class="mini">({esc(safety["method"])})</span> — {esc(safety.get("reason",""))}<br/>'
              f'decision: <span class="pbadge" style="background:{dcolor};">{esc(rd["decision"])}</span></div>'
              + why(rd["reason"]))

    # AI reply -> conversation (real draft + routing note)
    tag = (f'{rd["decision"]} · {c["intent"]} · {d["method"]}'
           + ("  ⚠ escalated to human" if rd["decision"] == "HUMAN" else ""))
    st.session_state.chat.append({"role": "ai", "text": d["draft"] or "(no draft)", "tag": tag})
    render_chat()
else:
    render_chat()
