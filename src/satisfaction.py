"""Customer-dissatisfaction detection — explicit + implicit (spec Steps 2/3).

Mirrors the dual-path discipline of pipeline.detect_safety: an LLM verifier when a key is present
(reads the message + previous AI reply + short history), else a dependency-free NLP rule detector.
The method field always states which path ran.

Crucial distinction (Step 3): a bare technical complaint ("the app keeps crashing") is NOT
dissatisfaction. Dissatisfaction requires a rejection/complaint-about-the-help pattern, an explicit
negative/human signal, or a repeated unresolved question.
"""
from __future__ import annotations

import re

from . import config, llm
from .preprocess import clean_for_embedding

# Whole-phrase rejection / dissatisfaction patterns (regex, case-insensitive).
_REJECTION_PATTERNS = [
    r"not what i (asked|meant|wanted)",
    r"(is|are)n'?t helping", r"(this|that) (still )?(does|doesn'?t|did not|didn't) (work|help)",
    r"not (understanding|listening|helping) me", r"you keep saying", r"same (thing|answer)",
    r"already (told|said|explained)", r"(answer|response|reply) is wrong", r"that'?s wrong",
    r"wrong answer", r"didn'?t (help|work|answer)", r"still (not|doesn'?t|isn'?t)",
    r"useless", r"unacceptable",
]
# Explicit request-for-human patterns.
_HUMAN_PATTERNS = [
    r"talk to (a |an )?(human|person|agent|someone|representative)",
    r"(connect|transfer) me", r"real (person|human|agent)", r"speak to (a |an )?(human|person|agent)",
    r"want (a |an )?(human|agent|person)", r"customer service rep",
]

_REJECTION_RE = [re.compile(p, re.I) for p in _REJECTION_PATTERNS]
_HUMAN_RE = [re.compile(p, re.I) for p in _HUMAN_PATTERNS]


# small stoplist so shared-content-word counting isn't fooled by common glue words
_STOP = {"the", "you", "and", "your", "not", "did", "for", "was", "are", "its", "her", "him",
         "she", "they", "them", "this", "that", "with", "have", "has", "had", "but", "get",
         "got", "why", "how", "can", "i've", "i'm", "when", "what", "who", "still", "keep"}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(clean_for_embedding(a).split()), set(clean_for_embedding(b).split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def _shared_content(a: str, b: str) -> set:
    """Content words (len>=3, non-stop) shared by two messages — signals the SAME issue restated."""
    ta = {t for t in clean_for_embedding(a).split() if len(t) >= 3 and t not in _STOP}
    tb = {t for t in clean_for_embedding(b).split() if len(t) >= 3 and t not in _STOP}
    return ta & tb


def _is_repeat(message: str, prior: str) -> bool:
    """A repeat if the two messages are broadly similar (jaccard) OR share >=3 content words.
    Pure jaccard under-fires on a longer restatement ('I already told you the app keeps crashing
    ...' vs 'the app keeps crashing ...'), so we also count shared content words."""
    return _jaccard(message, prior) >= config.REPEAT_SIM_THRESHOLD or len(_shared_content(message, prior)) >= 3


def _detect_rules(message: str, history, explicit_feedback, prev_ai_reply) -> dict:
    signals, score = [], 0.0
    if explicit_feedback == "helpful":
        return {"dissatisfied": False, "score": 0.0, "signals": ["explicit_helpful"], "method": "rules"}
    if explicit_feedback == "human_request":
        signals.append("explicit_human_request"); score = 1.0
    elif explicit_feedback == "not_helpful":
        signals.append("explicit_thumbs_down"); score = max(score, 0.6)

    if any(r.search(message) for r in _HUMAN_RE):
        signals.append("explicit_human_request"); score = 1.0
    if any(r.search(message) for r in _REJECTION_RE):
        signals.append("rejection_language"); score = max(score, 0.7)

    for prior in (history or []):
        if _is_repeat(message, prior):
            signals.append("repeated_unresolved_question"); score = max(score, 0.65)
            break

    dissatisfied = score >= config.DISSATISFACTION_THRESHOLD
    # de-dup signals, keep order
    seen, uniq = set(), []
    for s in signals:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return {"dissatisfied": dissatisfied, "score": round(score, 3), "signals": uniq, "method": "rules"}


def _detect_llm(message: str, history, explicit_feedback, prev_ai_reply) -> dict:
    # explicit signals are authoritative — short-circuit without spending a call
    if explicit_feedback == "helpful":
        return {"dissatisfied": False, "score": 0.0, "signals": ["explicit_helpful"], "method": "rules"}
    if explicit_feedback == "human_request":
        return {"dissatisfied": True, "score": 1.0, "signals": ["explicit_human_request"], "method": "rules"}
    hist = " | ".join((history or [])[-3:])
    system = (
        "You judge whether a customer is DISSATISFIED WITH THE SUPPORT HELP THEY RECEIVED — not "
        "whether they have a problem. Reporting an issue or asking for a refund is NOT "
        "dissatisfaction. A neutral technical description ('the app keeps crashing') is NOT "
        "dissatisfaction. Dissatisfaction ONLY when the customer rejects/criticises the AGENT'S help, "
        "repeats an unresolved issue after a reply, expresses frustration with the service response, "
        "or asks for a human. If there is no previous AI reply to react to (first contact), answer "
        "dissatisfied=0. Read the previous AI reply and history for context.\n"
        'Respond ONLY with JSON: {"dissatisfied": 0 or 1, "score": <0-1 float>, '
        '"reason": "<one short clause>"}.'
    )
    user = f'Previous AI reply: "{prev_ai_reply}"\nHistory: "{hist}"\nCurrent message: "{message}"'
    from .pipeline import _loads
    data = _loads(llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                           temperature=0.0, max_tokens=400, json_mode=True))
    dissatisfied = bool(int(data.get("dissatisfied", 0)))
    score = float(data.get("score", 0.0))
    if explicit_feedback == "not_helpful":
        score = max(score, 0.6); dissatisfied = dissatisfied or score >= config.DISSATISFACTION_THRESHOLD
    sig = [str(data.get("reason", "")).strip()] if data.get("reason") else []
    return {"dissatisfied": dissatisfied, "score": round(max(0.0, min(1.0, score)), 3),
            "signals": sig or (["dissatisfied"] if dissatisfied else []), "method": f"llm:{llm.last_provider()}"}


def detect_dissatisfaction(message: str, history=None, explicit_feedback=None,
                           prev_ai_reply: str = "", mode: str = "auto") -> dict:
    if mode == "rules":
        return _detect_rules(message, history, explicit_feedback, prev_ai_reply)
    if mode == "llm":
        return _detect_llm(message, history, explicit_feedback, prev_ai_reply)
    if llm.llm_available():
        try:
            return _detect_llm(message, history, explicit_feedback, prev_ai_reply)
        except Exception as e:
            out = _detect_rules(message, history, explicit_feedback, prev_ai_reply)
            out["method"] = f"rules (llm_error: {type(e).__name__})"
            return out
    return _detect_rules(message, history, explicit_feedback, prev_ai_reply)
