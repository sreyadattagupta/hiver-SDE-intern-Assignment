"""Human Support dashboard — the admin side of the human-in-the-loop loop.

Reads/writes the SAME on-disk conversation store as the customer console (app.py), so a conversation
escalated by a customer appears here for a human agent: full context, live timeline, a reply box, and
a Resolve & Verify action that writes a HUMAN-VERIFIED resolution into the RAG memory
(src/memory.py) for future retrieval. Only verified resolutions become trusted knowledge — raw
feedback never does.
"""
import streamlit as st

from src import memory, feedback_log
from src import conversation_store as cs
from src import ui_components as ui
from src.ui_components import esc, md

st.set_page_config(page_title="Uber Support AI — Human Support", page_icon="👤", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(ui.CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🚕 Uber Support AI")
    st.caption("Agent side. Use the page nav above to switch back to the **Customer Console**. "
               "Conversations escalated by customers appear in the Waiting queue below.")

store = cs.ConversationStore()

if "admin_sel" not in st.session_state:
    st.session_state.admin_sel = None
if "admin_queue" not in st.session_state:
    st.session_state.admin_queue = cs.WAITING_FOR_HUMAN
if "admin_live" not in st.session_state:
    st.session_state.admin_live = True

# ----------------------------- top bar -----------------------------
counts = store.counts_by_status()
fb_stats = feedback_log.feedback_stats()
n_mem = len(memory.load_resolutions())
ui.render_top_bar(
    "👤 Human Support Dashboard",
    f'<span class="dot hum"></span><span class="admintag">Agent Console</span><br/>'
    f'<b>Waiting:</b> {counts.get(cs.WAITING_FOR_HUMAN,0)} · '
    f'<b>Active:</b> {counts.get(cs.HUMAN_ACTIVE,0)} · '
    f'<b>Resolved:</b> {counts.get(cs.RESOLVED,0)} · '
    f'<b>Verified memory:</b> {n_mem}')

QUEUES = {
    "Waiting": cs.WAITING_FOR_HUMAN,
    "Active": cs.HUMAN_ACTIVE,
    "Resolved": cs.RESOLVED,
    "All": None,
}
labels = [f"{name} ({counts.get(stt,0) if stt else sum(counts.values())})"
          for name, stt in QUEUES.items()]
choice = st.radio("Queue", labels, horizontal=True, label_visibility="collapsed")
sel_name = list(QUEUES.keys())[labels.index(choice)]
st.session_state.admin_queue = QUEUES[sel_name]

col_c1, col_c2, col_c3 = st.columns([1, 1, 6])
st.session_state.admin_live = col_c1.toggle("Live", value=st.session_state.admin_live,
                                            help="Auto-refresh the queue")
if col_c2.button("🔄 Refresh"):
    st.rerun()

left, right = st.columns([1, 2.2], gap="large")


# ----------------------------- conversation list (left) -----------------------------
def render_list():
    convs = store.list_conversations(status=st.session_state.admin_queue)
    if not convs:
        md('<div class="emptead">No conversations in this queue.</div>')
        return
    for conv in convs:
        cid = conv["conversation_id"]
        first_customer = next((m["text"] for m in conv["messages"] if m["role"] == cs.CUSTOMER),
                              "(no message)")
        md(f'<div class="conv-card"><div class="q">{esc(first_customer[:60])}</div>'
           f'<div class="m">{ui.status_badge(conv["status"])} · {esc(conv.get("intent") or "—")} · '
           f'{esc(cid[-6:])}</div></div>')
        if st.button("Open", key=f"open_{cid}", use_container_width=True):
            st.session_state.admin_sel = cid
            st.rerun()


with left:
    st.markdown('<div class="panel-h">📥 Queue</div>', unsafe_allow_html=True)
    list_box = st.container(height=480, border=False)
    if st.session_state.admin_live:
        @st.fragment(run_every=3)
        def _poll_list():
            with list_box:
                render_list()
        _poll_list()
    else:
        with list_box:
            render_list()


# ----------------------------- conversation detail (right) -----------------------------
def render_detail():
    cid = st.session_state.admin_sel
    if cid is None:
        md('<div class="emptead">Select a conversation from the queue to view its full context.</div>')
        return
    conv = store.get(cid)
    if conv is None:
        md('<div class="emptead">Conversation not found.</div>')
        return

    md(f'<div class="chat-h"><span class="t">Conversation</span>'
       f'<span class="s">{esc(cid)}</span></div>')
    ev = conv.get("retrieved_evidence") or []
    md('<div class="info-grid">'
       f'<span class="k">Status</span><span>{ui.status_badge(conv["status"])}</span>'
       f'<span class="k">Intent</span><span>{esc(conv.get("intent") or "—")}</span>'
       f'<span class="k">Safety</span><span>{esc(conv.get("safety_status","no"))}</span>'
       f'<span class="k">Created</span><span>{esc(conv.get("created_at","")[:19])}</span>'
       f'<span class="k">Escalation</span><span>{esc(conv.get("escalation_reason") or "—")}</span>'
       f'<span class="k">Evidence</span><span>{esc(", ".join(str(e.get("pair_id")) for e in ev) or "—")}</span>'
       '</div>')

    tl = st.container(height=300, border=True)
    with tl:
        for m in conv["messages"]:
            ui.render_message(m)


def render_controls():
    cid = st.session_state.admin_sel
    if cid is None:
        return
    conv = store.get(cid)
    if conv is None:
        return
    status = conv["status"]

    if status in (cs.WAITING_FOR_HUMAN, cs.HUMAN_ACTIVE):
        reply = st.text_area("Reply to customer", key=f"reply_{cid}", height=80,
                             placeholder="Type your reply to the customer…")
        if st.button("➤ Send reply", key=f"send_{cid}", type="primary"):
            if reply.strip():
                store.add_message(cid, cs.HUMAN, reply.strip())
                st.rerun()
            else:
                st.warning("Enter a reply first.")

    if status in (cs.WAITING_FOR_HUMAN, cs.HUMAN_ACTIVE):
        with st.expander("✅ Resolve & Verify (writes human-verified resolution to RAG memory)",
                         expanded=False):
            resolution = st.text_area("Final resolution", key=f"res_{cid}",
                                      placeholder="e.g. Confirmed a duplicate charge; the duplicate "
                                                  "will be refunded per policy within 3–5 days.")
            category = st.text_input("Category (optional)", key=f"cat_{cid}")
            note = st.text_input("Internal note (optional, not shown to customer)", key=f"note_{cid}")
            if st.button("✅ Resolve & Verify", key=f"resolve_{cid}", type="primary"):
                if resolution.strip():
                    query = next((m["text"] for m in conv["messages"] if m["role"] == cs.CUSTOMER), "")
                    ai_resp = next((m["text"] for m in conv["messages"] if m["role"] == cs.AI), "")
                    rec = memory.build_record(
                        original_query=query, human_resolution=resolution.strip(),
                        intent=conv.get("intent", ""), escalation_reason=conv.get("escalation_reason", ""),
                        ai_response=ai_resp, category=category.strip(), internal_note=note.strip())
                    out = memory.add_resolution(rec)
                    store.set_resolution(cid, resolution=resolution.strip(), category=category.strip(),
                                         note=note.strip(), resolution_id=out["resolution_id"])
                    conflict = "  ⚠ potential knowledge conflict flagged" if out.get("conflict_flag") else ""
                    st.success(f"Verified resolution stored ({out['status']}, id={out['resolution_id']})"
                               f"{conflict}. It is now retrievable for similar future questions.")
                    st.rerun()
                else:
                    st.warning("Enter a resolution before verifying.")

    if status == cs.RESOLVED:
        res = conv.get("resolution") or {}
        md('<div class="resolved-banner">✅ <b>Resolved.</b><br/>'
           f'{esc(res.get("text",""))}<br/><span class="mini">id={esc(res.get("resolution_id",""))} · '
           f'category={esc(res.get("category") or "—")}</span></div>')


with right:
    detail_box = st.container(border=False)
    if st.session_state.admin_live and st.session_state.admin_sel is not None:
        @st.fragment(run_every=3)
        def _poll_detail():
            with detail_box:
                render_detail()
        _poll_detail()
    else:
        with detail_box:
            render_detail()
    # controls live OUTSIDE the timed fragment so typing a reply is never interrupted by a refresh
    render_controls()
