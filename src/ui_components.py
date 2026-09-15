"""Shared Streamlit UI helpers + CSS for the customer console and the human-support dashboard.

Kept here so both pages share one visual language and one set of escaping/rendering helpers, and so
app.py stays a thin viewer (SPEC.md 0.1). No agent logic lives here — only presentation.
"""
from __future__ import annotations

import html

import streamlit as st

from src import conversation_store as cs

INTENT_COLORS = {
    "billing_payment": "#2563eb", "account_access": "#7c3aed", "trip_issue": "#0891b2",
    "safety_incident": "#dc2626", "delivery_order": "#ea580c", "service_complaint": "#db2777",
    "general_query": "#64748b", "": "#64748b",
}

STATUS_META = {
    cs.AI_ACTIVE:        ("AI ACTIVE", "#2563eb"),
    cs.WAITING_FOR_HUMAN:("WAITING FOR HUMAN", "#dc2626"),
    cs.HUMAN_ACTIVE:     ("HUMAN ACTIVE", "#d97706"),
    cs.RESOLVED:         ("RESOLVED", "#16a34a"),
    cs.CLOSED:           ("CLOSED", "#64748b"),
}


def esc(x) -> str:
    return html.escape(str(x))


def md(h: str) -> None:
    st.markdown("\n".join(l.strip() for l in h.splitlines() if l.strip()), unsafe_allow_html=True)


def status_badge(status: str) -> str:
    label, color = STATUS_META.get(status, (status, "#64748b"))
    return f'<span class="pbadge" style="background:{color};">{esc(label)}</span>'


CSS = """
<style>
header[data-testid="stHeader"] {display:none;}
#MainMenu, footer {visibility:hidden;}
.block-container {max-width: 1400px; padding-top: 1.0rem; padding-bottom: .6rem; overflow-x:hidden;}
.bub, .pstage .kv, .pstage .top {overflow-wrap:anywhere; word-break:break-word;}
.pstage {overflow:hidden;}
/* ---------- top bar (never clipped: our own bar, Streamlit header hidden) ---------- */
.topbar {background:linear-gradient(90deg,#0b0f19 0%,#111827 60%,#1f2937 100%);
         border-radius:14px; padding:14px 22px; color:#fff; display:flex; align-items:center;
         justify-content:space-between; margin-bottom:14px; gap:16px; flex-wrap:wrap;}
.topbar .brand {font-size:1.25rem; font-weight:700; letter-spacing:-.3px;}
.topbar .meta {font-size:.78rem; color:#cbd5e1; text-align:right; line-height:1.5;}
.topbar .meta b {color:#e5e7eb;}
.dot {height:9px; width:9px; border-radius:50%; display:inline-block; margin-right:6px;}
.dot.on {background:#22c55e; box-shadow:0 0 0 3px rgba(34,197,94,.25);}
.dot.hum {background:#f59e0b; box-shadow:0 0 0 3px rgba(245,158,11,.25);}
.online {color:#22c55e; font-weight:600;}
.admintag {color:#f59e0b; font-weight:600;}
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
@media (prefers-color-scheme: dark){ .pwhy{color:#94a3b8;} .pwhy .lb{color:#e5e7eb;} }
.pbadge {display:inline-block; padding:1px 8px; border-radius:6px; color:#fff; font-size:.72rem; font-weight:600;}
.parrow {text-align:center; color:#cbd5e1; font-size:.9rem; margin:1px 0;}
.mini {font-size:.7rem; color:#94a3b8;}
.evrow {font-size:.7rem; color:#94a3b8; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.mbar {height:6px; border-radius:5px; background:rgba(128,128,128,.2); overflow:hidden; margin-top:4px;}
.mbar > span {display:block; height:100%;}
/* ---------- chat ---------- */
.chat-h {display:flex; align-items:baseline; gap:10px; border-bottom:1px solid rgba(128,128,128,.18);
         padding-bottom:8px; margin-bottom:6px;}
.chat-h .t {font-size:1.15rem; font-weight:700;}
.chat-h .s {font-size:.8rem; color:#64748b;}
.chatwrap {padding:2px 6px 2px 2px;}
.row {display:flex; margin:9px 0;}
.row.cust {justify-content:flex-start;}
.row.ai, .row.human {justify-content:flex-end;}
.row.system {justify-content:center;}
.who {font-size:.68rem; font-weight:700; letter-spacing:.05em; color:#94a3b8; margin:0 6px 3px;}
.bub {max-width:76%; padding:11px 15px; border-radius:16px; font-size:.95rem; line-height:1.45;}
.bub.cust {background:#f1f5f9; color:#0f172a; border:1px solid #e2e8f0; border-bottom-left-radius:5px;}
.bub.ai {background:#2563eb; color:#fff; border-bottom-right-radius:5px;}
.bub.human {background:#0e7490; color:#fff; border-bottom-right-radius:5px;}
.bub.system {background:transparent; color:#94a3b8; font-size:.76rem; font-style:italic;
             border:1px dashed rgba(128,128,128,.35); padding:5px 12px; max-width:88%; text-align:center;}
.bub .tag {display:block; font-size:.7rem; opacity:.75; margin-top:6px;}
.ts {font-size:.62rem; color:#94a3b8; margin:2px 6px 0;}
.emptead {color:#94a3b8; text-align:center; padding:40px 10px; font-size:.9rem;}
@media (prefers-color-scheme: dark){
  .bub.cust{background:#1e293b; color:#e2e8f0; border-color:#334155;}
  .pstage .kv{color:#94a3b8;} .pstage .kv b{color:#e5e7eb;}
}
:root[data-theme="dark"] .bub.cust{background:#1e293b; color:#e2e8f0; border-color:#334155;}
/* ---------- feedback / escalation / human console / loop viz ---------- */
.fbnote {font-size:.72rem; color:#64748b; text-align:right; margin-top:2px;}
.esc-banner {border:1px solid #dc2626; background:rgba(220,38,38,.07); border-radius:11px;
             padding:10px 13px; margin:8px 0; color:#b91c1c; font-size:.85rem; line-height:1.5;}
.esc-banner b {color:#dc2626;}
.wait-banner {border:1px solid #d97706; background:rgba(245,158,11,.08); border-radius:11px;
              padding:10px 13px; margin:8px 0; color:#b45309; font-size:.88rem; line-height:1.5;}
.resolved-banner {border:1px solid #16a34a; background:rgba(22,163,74,.08); border-radius:11px;
                  padding:10px 13px; margin:8px 0; color:#15803d; font-size:.88rem; line-height:1.5;}
.loopviz {display:flex; flex-wrap:wrap; gap:6px; align-items:center; font-size:.72rem;
          color:#475569; margin:8px 0; padding-top:6px; border-top:1px dashed rgba(128,128,128,.28);}
.loopviz .n {background:rgba(37,99,235,.1); border-radius:6px; padding:2px 7px; white-space:nowrap;}
.loopviz .n.ok {background:rgba(22,163,74,.12); color:#15803d;}
/* ---------- admin ---------- */
.conv-card {border:1px solid rgba(128,128,128,.22); border-radius:10px; padding:8px 11px; margin-bottom:6px;}
.conv-card .q {font-size:.82rem; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;}
.conv-card .m {font-size:.68rem; color:#94a3b8; margin-top:3px;}
.info-grid {display:grid; grid-template-columns:auto 1fr; gap:2px 12px; font-size:.78rem; margin:6px 0;}
.info-grid .k {color:#94a3b8;}
</style>
"""


def render_top_bar(brand_html: str, meta_html: str) -> None:
    md(f'<div class="topbar"><div class="brand">{brand_html}</div><div class="meta">{meta_html}</div></div>')


def _fmt_ts(ts: str) -> str:
    # ISO -> HH:MM (UTC); tolerate bad input
    try:
        return ts[11:16]
    except Exception:
        return ""


def render_message(msg: dict) -> None:
    """Render one timeline message as a role-styled bubble (customer/ai/human/system)."""
    role = msg.get("role")
    text = esc(msg.get("text", ""))
    ts = _fmt_ts(msg.get("ts", ""))
    if role == cs.CUSTOMER:
        md(f'<div class="row cust"><div><div class="who">Customer</div>'
           f'<div class="bub cust">{text}</div><div class="ts">{ts}</div></div></div>')
    elif role == cs.AI:
        tag = msg.get("meta", {}).get("tag", "")
        tag_html = f'<span class="tag">{esc(tag)}</span>' if tag else ""
        md(f'<div class="row ai"><div><div class="who" style="text-align:right;">AI Support Agent</div>'
           f'<div class="bub ai">{text}{tag_html}</div><div class="ts" style="text-align:right;">{ts}</div></div></div>')
    elif role == cs.HUMAN:
        md(f'<div class="row human"><div><div class="who" style="text-align:right;">👤 Human Support</div>'
           f'<div class="bub human">{text}</div><div class="ts" style="text-align:right;">{ts}</div></div></div>')
    else:  # system
        md(f'<div class="row system"><div class="bub system">🛈 {text}</div></div>')
