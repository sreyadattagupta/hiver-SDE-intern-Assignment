# Human-in-the-Loop Support — QA Report

**Scope:** a real customer↔AI↔human support platform built on top of the existing 4-stage agent, with
a shared file-backed backend, a separate Human Support dashboard, an explicit conversation state
machine, feedback-driven escalation, and human-verified-resolution RAG memory.

**Method:** static analysis (pyflakes) + unit tests + headless end-to-end tests through the real
Streamlit button handlers (`streamlit.testing.v1.AppTest`) + **live Chrome/Playwright** driving the
running app. Evidence = the numbered screenshots in `docs/screenshots/qa_*.png` and the test suite.

**Environment:** Windows 11 · Python 3.12.7 · Streamlit 1.63 · LLM chain live (`groq → gemini`) during
the browser run, so the pipeline values below are real model output, not the offline path.

---

## Result summary

| Area | Verdict | Evidence |
|---|---|---|
| Implementation | **PASS** | new modules below; 66/66 tests |
| Customer chat | **PASS** | `qa_02`, `qa_04` — real 4-stage pipeline, bubbles, timestamps |
| Helpful feedback | **PASS** | `test_e2e_flow::test_helpful_keeps_ai_and_logs_feedback` — logged, stays AI |
| Not Helpful → Human | **PASS** | `qa_03` — status `WAITING_FOR_HUMAN`, "Connecting you…" banner |
| Admin dashboard | **PASS** | `qa_05` — queues, full context, escalation reason, evidence IDs |
| Real-time messaging | **PASS** | `qa_06` — customer tab received the human reply via polling, no manual refresh |
| Human resolution | **PASS** | `qa_07` — status `RESOLVED`, resolution persisted |
| Knowledge pipeline | **PASS** | `data/resolution_memory.jsonl` grew by exactly 1 verified record |
| Future RAG retrieval | **PASS** | `qa_04` — a verified resolution surfaced first (sim 0.921, "1 verified") and grounded the draft |
| Negative-knowledge guard | **PASS** | unresolved escalation never entered memory (`test_unresolved_conversation_never_enters_memory`; delivery convo left in Waiting) |
| Existing AI pipeline | **PASS** | original 49 tests green; classify/retrieve/draft/verify/safety/route unchanged |
| Browser testing | **PASS** | full chain driven in Chrome across two tabs |
| Regression | **PASS** | `pytest -q` → **66 passed** (49 original + 12 store + 5 e2e) |
| Responsive (1366×768) | **PASS** | `qa_09` — header fully visible, panels independent, no overlap |

---

## What was built (integration points, not a rewrite)

- `src/conversation_store.py` — shared, file-backed store (one atomic JSON per conversation) + guarded
  state machine `AI_ACTIVE → WAITING_FOR_HUMAN → HUMAN_ACTIVE → RESOLVED (→ CLOSED)`.
- `src/feedback_log.py` — raw feedback tier (`data/feedback_log.jsonl`). Raw → validation → **only**
  human-verified resolutions become trusted knowledge. 👍 never auto-enters RAG memory.
- `src/ui_components.py` — shared CSS + role-styled timeline (customer/ai/human/system).
- `app.py` — customer console rewired to the store; live 4-stage pipeline; 👍/👎/💬; polling
  (`st.fragment(run_every=3)`) so human replies arrive without a refresh.
- `pages/1_Human_Support.py` — Human Support dashboard: Waiting/Active/Resolved/All queues, full
  timeline + escalation context, human reply, **Resolve & Verify** → `memory.add_resolution`.
- Reused unchanged: `pipeline.py`, `escalation.py`, `satisfaction.py`, `memory.py`,
  `resolution_retrieval.py`, `retrieval.py`, `config.py`.

## Tests executed

- `tests/test_conversation_store.py` (12) — CRUD, state-machine transitions, invalid-transition
  guard, escalation idempotency, human-reply transition, resolution, reopen-on-new-message,
  feedback idempotency, raw feedback tier.
- `tests/test_e2e_flow.py` (5) — helpful path, not-helpful escalation, **full loop** (customer →
  AI → 👎 → escalation → human reply → resolve & verify → memory → future retrieval), negative
  knowledge, double-click idempotency — all through the real AppTest button handlers.
- Full suite: **66 passed** in ~95s. `pyflakes` clean on all new files.

## Bugs found & fixed during QA

| # | Sev | Issue | Root cause | Fix | Regression guard |
|---|---|---|---|---|---|
| 1 | P1 | Send crashed with `StreamlitWidgetAlreadyInstantiatedError` | wrote `st.session_state.msg=""` after the `msg` widget was instantiated | clear via a pre-widget `_clear_msg` flag | `test_e2e_flow` (send path) |
| 2 | P2 | No navigation to the admin page; `/Human_Support` 404'd assets | Streamlit process was started stale before `pages/` was registered; hidden header removed the sidebar toggle | restart picked up `pages/`; added `initial_sidebar_state="expanded"` + sidebar caption | manual browser nav verified (`qa_05`) |
| 3 | P3 | `st.page_link("app.py")` raised `StreamlitPageNotFoundError` | entrypoint not addressable via `page_link` in this Streamlit version | removed explicit links; rely on Streamlit's automatic multipage nav | browser nav verified |

## Honest limitations / NOT claimed

- **No authentication** between the customer console and the admin dashboard — the Human Support page
  is a separate route, not access-controlled. Fine for a single-operator prototype; a real deployment
  needs auth. (Documented, not hidden.)
- **Concurrency:** writes are in-process-locked and each save is atomic, so the realistic
  single-customer / single-agent-per-conversation pattern is safe. Two processes mutating the **same**
  conversation simultaneously is last-writer-wins on that file — acceptable for a prototype.
- **NOT TESTABLE this session — LLM-failure fallback in the browser:** a live API key was present, so
  the browser run used the real LLM. The rules/offline fallback path is covered by the original unit
  tests (`test_pipeline.py`) and the deterministic e2e run (no key), not re-forced in-browser.
- **NOT load-tested:** many simultaneous customers / high write throughput not benchmarked.
- **Lexical paraphrase ceiling** on resolution retrieval remains (TF-IDF; sentence-transformers is
  sandbox-gated) — inherited from the existing memory layer, see `reports/failure_analysis_feedback.md`.

## Verdict

The critical chain — **customer → AI → Not Helpful → automatic escalation → admin queue → human reply
→ customer receives it → human resolution → verified memory → future retrieval** — genuinely works,
verified in a real browser and by tests. Not a demo where buttons change colour.
