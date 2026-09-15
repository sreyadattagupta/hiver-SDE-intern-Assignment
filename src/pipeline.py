"""
The real Uber_Support AI agent. Nothing here is mocked or canned (see CLAUDE.md 0.1).

Public contracts (stable — eval.py, app.py, routing all depend on these):

    classify_intent(message) -> {"intent", "confidence", "runner_up", "method"}
    retrieve_similar(message, k=3) -> [{"pair_id","customer_msg","brand_reply","score"}, ...]
    draft_reply(message, retrieved) -> {"draft","cited_example_id","justification","method"}
    route_decision(classification, message, thread_history) -> {"decision","reason"}

The "method" field is added to classify/draft so every output HONESTLY states whether the
real LLM path or the rule-based fallback produced it. It is never hidden.

The 7 intents (derived from reading real Uber_Support inbound tweets):
    billing_payment  - charges, refunds, credits, promo codes, penalties, "why did it cost X"
    account_access   - login problems, hacked account, app reinstall, account settings/VIP
    trip_issue       - ride/driver problems: cancellation, wrong route, pickup, lateness, rating
    safety_incident  - physical safety: accident, assault, unsafe/drunk driver, threats, harassment
    delivery_order   - UberEats: missing/wrong/cancelled order, sold out, food charged not delivered
    service_complaint- general dissatisfaction, unresolved tickets, "no response", rants, "sue you"
    general_query    - informational how-to / where questions, follow-ups, "check DM"
"""
from __future__ import annotations

import json
import re
import os

from . import llm

INTENTS = [
    "billing_payment", "account_access", "trip_issue", "safety_incident",
    "delivery_order", "service_complaint", "general_query",
]

PAIRS_PATH = os.path.join("data", "uber_pairs.csv")


def _loads(raw: str) -> dict:
    """Parse a JSON object from an LLM response, tolerating code fences, prose, reasoning text,
    and stray braces. Scans for balanced {...} candidates and returns the first that parses."""
    raw = (raw or "").strip()
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    # balanced-brace scan: try each '{' as a start, find its matching '}', attempt to parse
    for start in (i for i, ch in enumerate(raw) if ch == "{"):
        depth = 0
        for end in range(start, len(raw)):
            if raw[end] == "{":
                depth += 1
            elif raw[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:end + 1])
                    except Exception:
                        break  # not valid from this start; try next '{'
    raise ValueError(f"no JSON object in LLM output: {raw[:200]!r}")

# --- Few-shot examples for the LLM classifier (2-3 real-style examples per intent) ---
FEWSHOT = [
    ("why did my uber cost 7.50 when the app quoted 4-6? charge me the wrong amount", "billing_payment"),
    ("Order accepted but I was charged twice, how do I get a refund for the duplicate", "billing_payment"),
    ("promo code DELNOV75 is not getting applied to my final price", "billing_payment"),
    ("i reset my phone and reinstalled the app, now i cant log back into my account", "account_access"),
    ("my Uber account was hacked more than 36 hours ago and I cant get back in", "account_access"),
    ("my driver cancelled the ride because they missed the turn two minutes from me", "trip_issue"),
    ("driver took a very long way and my trip cost way more than it should", "trip_issue"),
    ("why do I have to walk to a pickup point, isnt the driver supposed to come to me", "trip_issue"),
    ("our uber driver just offered us a shot of fireball while driving, this is unsafe", "safety_incident"),
    ("the driver stopped the cab and asked me to get off on the highway, I felt threatened", "safety_incident"),
    ("our UberEats order was rejected last night but my card was still charged", "delivery_order"),
    ("order accepted by restaurant but details show it as sold out, my food never came", "delivery_order"),
    ("day 4 and still no real response, worst customer service I have ever dealt with", "service_complaint"),
    ("its been 2 hours trying to reach you and nobody is responding, this is unacceptable", "service_complaint"),
    ("is there a number I can call to reach a real person", "general_query"),
    ("Check DM please", "general_query"),
]

# --- Rule-based keyword scores per intent (transparent baseline + zero-key fallback) ---
_RULES = {
    "billing_payment": ["charge", "charged", "refund", "credit", "cost", "price", "promo", "code",
                        "coupon", "payment", "pay ", "penalty", "fare", "money back", "billed", "receipt",
                        "overcharg", "dispute", "invoice"],
    "account_access": ["log in", "login", "log back", "sign in", "password", "hacked", "account was",
                       "cant access", "can't access", "locked out", "reinstall", "reset my phone", "verify",
                       "otp", "my account"],
    "trip_issue": ["driver cancel", "cancelled my", "cancel the ride", "canceled", "pickup", "pick up",
                  "long way", "wrong route", "late", "rating", "stars", "gps", "trip", "ride", "waiting for",
                  "no show", "didnt show", "didn't show"],
    "safety_incident": ["unsafe", "assault", "accident", "harass", "threat", "drunk", "kicked me out",
                        "asked me to get off", "hit a", "hitting", "grope", "weapon", "danger", "attacked",
                        "crash", "collision", "reckless"],
    "delivery_order": ["ubereats", "uber eats", "eats", "order", "food", "restaurant", "delivery",
                      "delivered", "sold out", "rejected", "courier", "meal"],
    "service_complaint": ["worst", "terrible", "unacceptable", "no response", "not responding", "sue",
                         "ridiculous", "disgusting", "never received such", "day 4", "days and", "still no",
                         "useless", "waste", "fed up", "complain"],
    "general_query": ["how do i", "how can i", "where can i", "is there a number", "check dm", "dm",
                     "any way to", "can i", "what is", "follow up", "?"],
}

# Legacy naive substring list — kept ONLY for the before/after comparison in the report/demo.
# route_decision no longer uses this; it uses the context-aware detect_safety() below.
_NAIVE_SAFETY_KEYWORDS = ["unsafe", "assault", "accident", "harass", "threat", "drunk driver",
                          "attacked", "weapon", "groped", "kidnap", "crash", "collision"]

_FRUSTRATION_WORDS = ["angry", "furious", "worst", "terrible", "unacceptable", "ridiculous",
                     "disgusting", "hate", "fuck", "shit", "wtf", "sue", "useless", "pathetic",
                     "never again", "fed up", "!!!"]

# ---------- Context-aware safety detection (NLP: whole-word + context windows + negation) ----------
# Unambiguous physical-safety terms: a whole-word hit is a safety signal unless negated.
_SAFETY_STRONG = {
    "assault", "assaulted", "assaulting", "harass", "harassed", "harassing", "harassment",
    "groped", "grope", "groping", "molest", "molested", "kidnap", "kidnapped", "weapon",
    "knife", "raped", "rape", "stalked", "stalking", "threatened", "threatening",
}
# Ambiguous terms: safety ONLY in a physical/vehicle context, NOT in a tech/app context.
# term -> (physical_context_words, disqualifying_tech_context_words)
_SAFETY_AMBIGUOUS = {
    "crash":     ({"car", "vehicle", "cab", "driver", "road", "into", "hit", "highway", "truck", "bike", "auto"},
                  {"app", "application", "website", "site", "page", "screen", "software", "system",
                   "update", "keeps", "browser", "login", "load", "loading", "server"}),
    "crashed":   ({"car", "vehicle", "cab", "driver", "road", "into", "hit", "highway"},
                  {"app", "website", "site", "page", "screen", "software", "system", "server"}),
    "crashing":  ({"car", "vehicle", "driver", "road", "into"},
                  {"app", "application", "website", "site", "page", "screen", "software", "keeps", "server"}),
    "accident":  ({"car", "vehicle", "road", "hit", "rear", "rear-ended", "driver", "injured", "hospital", "collision"},
                  set()),
    "collision": ({"car", "vehicle", "road", "driver", "another"}, set()),
    "drunk":     ({"driver", "driving", "wheel", "cab"},
                  {"i", "we", "myself", "getting", "got"}),   # "I was drunk" is not a safety report
    "reckless":  ({"driver", "driving", "drove", "speeding", "road", "cab"}, set()),
    "dangerous": ({"driver", "driving", "road", "speed", "cab"}, set()),
    "unsafe":    (set(), set()),   # generic-but-safety; still subject to negation
    "attacked":  ({"driver", "passenger", "physically", "me"}, {"account", "spam", "app"}),
    "threat":    ({"driver", "me", "physical", "violence"}, set()),
}
_NEGATION_TOKENS = {"no", "not", "never", "without", "wasnt", "isnt", "didnt", "dont",
                    "avoid", "avoided", "prevent", "prevented", "almost", "nearly", "no-one"}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _negated(tokens: list[str], idx: int) -> bool:
    """True if a negation token appears in the 3 tokens before position idx."""
    return any(t.replace("'", "") in _NEGATION_TOKENS for t in tokens[max(0, idx - 3):idx])


def _detect_safety_rules(message: str) -> dict:
    """Context-aware safety detection without an LLM: whole-word match + ±window context + negation."""
    tokens = _tokenize(message)
    triggers, notes = [], []
    for i, tok in enumerate(tokens):
        if tok in _SAFETY_STRONG:
            if _negated(tokens, i):
                notes.append(f"'{tok}' ignored (negated)")
                continue
            triggers.append(tok)
        elif tok in _SAFETY_AMBIGUOUS:
            physical, tech = _SAFETY_AMBIGUOUS[tok]
            window = set(tokens[max(0, i - 4):i + 5])
            if tech & window:
                notes.append(f"'{tok}' ignored (tech context: {', '.join(sorted(tech & window))})")
                continue
            if physical and not (physical & window):
                notes.append(f"'{tok}' ignored (no physical/vehicle context nearby)")
                continue
            if _negated(tokens, i):
                notes.append(f"'{tok}' ignored (negated)")
                continue
            triggers.append(tok)
    if triggers:
        reason = f"physical-safety terms in context: {', '.join(sorted(set(triggers)))}"
        return {"is_safety": True, "triggers": sorted(set(triggers)), "reason": reason, "method": "rules"}
    reason = "no physical-safety signal" + (f" ({'; '.join(notes)})" if notes else "")
    return {"is_safety": False, "triggers": [], "reason": reason, "method": "rules"}


def _detect_safety_llm(message: str) -> dict:
    system = (
        "You are a safety triage filter for ride-hailing support messages. Decide if the message "
        "reports a PHYSICAL-SAFETY incident — a car accident/collision, assault, harassment, "
        "unsafe/reckless/drunk DRIVING, threats, or weapons. A software/app 'crash', a billing "
        "problem, or general anger are NOT physical-safety incidents. Read the full context.\n"
        'Respond ONLY with JSON: {"safety": 0 or 1, "trigger": "<short phrase or empty>", '
        '"reason": "<one clause>"}.'
    )
    data = _loads(llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f'Message: "{message}"'}],
        temperature=0.0, max_tokens=1000, json_mode=True,
    ))
    is_safety = bool(int(data.get("safety", 0)))
    trig = str(data.get("trigger", "")).strip()
    return {"is_safety": is_safety, "triggers": [trig] if trig else [],
            "reason": str(data.get("reason", "")).strip() or ("safety incident" if is_safety else "no safety signal"),
            "method": f"llm:{llm.last_provider()}"}


def detect_safety(message: str, mode: str = "auto") -> dict:
    """Context-aware safety check. Returns {is_safety, triggers, reason, method}.
    LLM verifier when a key is set (full-context understanding), else the NLP rule detector."""
    if mode == "rules":
        return _detect_safety_rules(message)
    if mode == "llm":
        return _detect_safety_llm(message)
    if llm.llm_available():
        try:
            return _detect_safety_llm(message)
        except Exception as e:
            out = _detect_safety_rules(message)
            out["method"] = f"rules (llm_error: {type(e).__name__})"
            return out
    return _detect_safety_rules(message)


def naive_safety_hits(message: str) -> list:
    """Legacy blind substring matcher — for the before/after demo only (shows the 'crash' bug)."""
    m = message.lower()
    return [kw for kw in _NAIVE_SAFETY_KEYWORDS if kw in m]


# =========================== CLASSIFICATION ===========================

def _classify_rules(message: str) -> dict:
    m = message.lower()
    scores = {intent: 0 for intent in INTENTS}
    for intent, kws in _RULES.items():
        for kw in kws:
            if kw in m:
                scores[intent] += 1
    # Context-aware override for safety: don't let a bare 'crash'/'accident' substring win.
    # Trust the NLP context detector instead (fixes "app keeps crashing" -> not safety).
    if _detect_safety_rules(message)["is_safety"]:
        scores["safety_incident"] += 3
    else:
        scores["safety_incident"] = 0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_score = ranked[0]
    runner_up = ranked[1][0]
    total = sum(scores.values())
    if top_score == 0:
        return {"intent": "general_query", "confidence": 0.30, "runner_up": ranked[1][0],
                "method": "rules",
                "reason": "no strong intent keywords matched — defaulted to general_query"}
    if top == "safety_incident" and _detect_safety_rules(message)["is_safety"]:
        reason = "context-aware safety detector flagged a physical-safety signal"
    else:
        hits = [kw for kw in _RULES.get(top, []) if kw in m][:4]
        reason = (f"matched {top} keyword(s): {', '.join(repr(k) for k in hits)}" if hits
                  else f"highest keyword score for {top}")
    confidence = round(min(0.95, 0.45 + 0.5 * (top_score / total)), 3) if total else 0.3
    return {"intent": top, "confidence": confidence, "runner_up": runner_up, "method": "rules",
            "reason": reason}


def _classify_llm(message: str) -> dict:
    examples = "\n".join(f'- "{t}" -> {lab}' for t, lab in FEWSHOT)
    system = (
        "You are an intent classifier for Uber customer-support tweets. "
        f"Classify each message into exactly one of these intents: {', '.join(INTENTS)}.\n"
        "Definitions:\n"
        "billing_payment: charges, refunds, credits, promo codes, fares, penalties.\n"
        "account_access: login, hacked, password, app reinstall, account settings.\n"
        "trip_issue: ride/driver problems - cancellation, route, pickup, lateness, ratings.\n"
        "safety_incident: physical safety - accident, assault, unsafe/drunk driver, threats, harassment.\n"
        "delivery_order: UberEats food orders - missing/wrong/cancelled, sold out, charged-not-delivered.\n"
        "service_complaint: general dissatisfaction, unresolved tickets, no response, rants.\n"
        "general_query: informational how-to/where questions, follow-ups, 'check DM'.\n\n"
        "Examples:\n" + examples + "\n\n"
        'Respond ONLY with JSON: {"intent": <one intent>, "confidence": <0-1 float>, '
        '"runner_up": <second most likely intent>, "reason": <one short sentence citing the '
        'specific words/phrase in THIS message that drove the choice>}. '
        "Base confidence on how clear the message is."
    )
    raw = llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f'Classify this message:\n"{message}"'}],
        temperature=0.0, max_tokens=1200, json_mode=True,
    )
    data = _loads(raw)
    intent = data.get("intent", "general_query")
    if intent not in INTENTS:
        intent = "general_query"
    runner_up = data.get("runner_up", "general_query")
    if runner_up not in INTENTS:
        runner_up = "general_query"
    conf = float(data.get("confidence", 0.5))
    reason = str(data.get("reason", "")).strip() or f"model classified this as {intent}"
    return {"intent": intent, "confidence": round(max(0.0, min(1.0, conf)), 3),
            "runner_up": runner_up, "method": f"llm:{llm.last_provider()}", "reason": reason}


CONF_LOW = 0.55   # below this -> needs_context (also drives routing escalation)


def _with_needs_context(out: dict) -> dict:
    """Add a calibrated 'needs_context' advisory (Phase 10/11) without breaking the core contract."""
    out["needs_context"] = out.get("confidence", 0.0) < CONF_LOW
    return out


def classify_intent(message: str, mode: str = "auto") -> dict:
    """Classify one message. mode: 'auto' (LLM if key else rules), 'llm', or 'rules'.
    Returns {"intent","confidence","runner_up","method","needs_context"} — 'method' states the real
    path used; 'needs_context' flags genuinely low-confidence cases for routing/clarification."""
    if mode == "rules":
        return _with_needs_context(_classify_rules(message))
    if mode == "llm":
        return _with_needs_context(_classify_llm(message))
    # auto
    if llm.llm_available():
        try:
            return _with_needs_context(_classify_llm(message))
        except Exception as e:
            out = _classify_rules(message)
            out["method"] = f"rules (llm_error: {type(e).__name__})"
            return _with_needs_context(out)
    return _with_needs_context(_classify_rules(message))


# =========================== RETRIEVAL ===========================

def retrieve_similar(message: str, k: int = 3, method: str = "auto", intent: str | None = None,
                     use_memory: bool = True, memory_path: str | None = None) -> list:
    """Top-k grounding evidence, merging two layers (spec Step 10):
      1. human-verified resolution memory (higher trust) — gated by similarity threshold + intent
      2. historical (customer_msg -> brand_reply) pairs from the 1500-pair corpus

    Verified items receive a small additive trust boost (config.VERIFIED_TRUST_BOOST) so an equally
    relevant verified resolution outranks an ordinary historical reply — but a much-more-relevant
    historical reply can still win. Returns the stable contract {pair_id, customer_msg, brand_reply,
    score} plus 'method', 'source' ('human_resolution'|'historical'), and 'verified'. When the memory
    is empty the result is identical to the historical-only retriever (back-compat)."""
    from . import retrieval, resolution_retrieval, config

    historical = retrieval.retrieve(message, k=k, method=method)
    for h in historical:
        h.setdefault("source", "historical")
        h.setdefault("verified", False)

    verified = []
    if use_memory:
        try:
            verified = resolution_retrieval.retrieve_resolutions(message, intent=intent, k=k, path=memory_path)
        except Exception:
            verified = []   # memory retrieval must never break the core pipeline

    if not verified:
        return historical

    merged = []
    for v in verified:
        item = dict(v)
        item["score"] = round(item["score"] + config.VERIFIED_TRUST_BOOST, 4)
        merged.append(item)
    merged.extend(historical)
    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:k]


# =========================== DRAFT GENERATION ===========================

_DM_MARKERS = ["dm", "direct message", "send us a", "private message", "shoot us"]


def is_dm_deflection(text: str) -> bool:
    """True if a reply is just the lazy 'please DM us' template (grounding-gaming check, CLAUDE.md 5)."""
    t = text.lower()
    has_dm = any(m in t for m in _DM_MARKERS)
    # short + DM ask + no concrete resolution content
    return has_dm and len(text.split()) <= 30




_URL_RE = re.compile(r"https?://\S+|t\.co/\S+")
_UNSUPPORTED_PROMISE = [
    "we have refunded", "we've refunded", "your refund has been processed", "we have credited",
    "we've credited", "we have issued", "we have processed your", "has been refunded",
    "we have banned", "we have fired", "we have deactivated the driver",
]


def verify_draft(message: str, draft: str, retrieved: list, intent: str) -> dict:
    """Deterministic groundedness/quality checks (Phase 13). Returns {ok, issues}.
    Hard fails (ok=False) = hallucinated URL or unsupported promise. Soft flags don't fail."""
    issues, ok = [], True
    if not draft or len(draft.split()) < 4:
        return {"ok": False, "issues": ["empty_or_too_short"]}
    evidence = " ".join(r["brand_reply"] for r in retrieved).lower()
    # hard fail: a URL that appears in the draft but in NONE of the retrieved evidence replies
    for url in _URL_RE.findall(draft):
        if url.lower() not in evidence:
            issues.append(f"invented_url:{url}"); ok = False
    # hard fail: promising an action already completed (can't be grounded from a public tweet)
    low = draft.lower()
    for p in _UNSUPPORTED_PROMISE:
        if p in low:
            issues.append(f"unsupported_promise:{p}"); ok = False
    # soft flags (do not fail, but reported)
    if is_dm_deflection(draft):
        issues.append("dm_deflection_soft")
    if intent == "safety_incident" and "safe" not in low and "sorry" not in low:
        issues.append("safety_tone_missing_soft")
    return {"ok": ok, "issues": issues}


def _draft_llm(message: str, retrieved: list, strict: bool = False) -> dict:
    top = retrieved[0]
    ctx = "\n".join(
        f'[{r["pair_id"]}] (sim={r["score"]}) customer: "{r["customer_msg"]}" -> brand replied: "{r["brand_reply"]}"'
        for r in retrieved
    )
    system = (
        "You are an Uber_Support agent. Draft ONE short, empathetic public reply to the customer, "
        "grounded in how Uber_Support handled the most similar past cases below.\n"
        "REQUIREMENTS:\n"
        "1. Directly acknowledge THIS customer's specific issue in concrete terms.\n"
        "2. Give a useful next step. Do NOT default to a bare 'please DM us' — only ask for a DM "
        "when private info is genuinely needed (account/billing/safety), and even then acknowledge "
        "the issue first.\n"
        "3. Never claim an action is already done (no 'we have refunded/credited/banned') — you "
        "cannot promise outcomes.\n"
        "4. Only include a link if it appears in the past cases below. Do not invent URLs.\n"
        "5. Under 280 characters, match Uber's real tone.\n\n"
        f"Most similar past cases:\n{ctx}\n\n"
        'Respond ONLY with JSON: {"draft": <reply>, "cited_example_id": <pair_id>, '
        '"justification": <why that example fits: topic/resolution/tone>}.'
    )
    if strict:
        system += ("\nThe previous draft failed verification (invented a link or promised a "
                   "completed action). Regenerate WITHOUT any invented URL or completed-action claim.")
    raw = llm.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f'Customer message:\n"{message}"'}],
        temperature=0.2 if strict else 0.3, max_tokens=2048, json_mode=True,
    )
    data = _loads(raw)
    return {
        "draft": data.get("draft", "").strip(),
        "cited_example_id": str(data.get("cited_example_id", top["pair_id"])),
        "justification": data.get("justification", "").strip(),
        "method": f"llm:{llm.last_provider()}",
    }


# Offline (no-LLM) response synthesizer — composes a FRESH reply from the 3 references instead of
# copying one verbatim, so the fallback still feels authored and grounded (never leaks a real
# customer name or a specific URL from the dataset).
_ACK = {
    "billing_payment": "So sorry about the trouble with your charge — that's never what we want.",
    "account_access": "Sorry you're having trouble getting into your account.",
    "trip_issue": "Sorry your trip didn't go the way it should have.",
    "safety_incident": "We're really sorry to hear this — your safety is our top priority.",
    "delivery_order": "Sorry your order didn't arrive the way it should have.",
    "service_complaint": "Sorry for the poor experience and for the wait to hear back.",
    "general_query": "Happy to help you with this.",
    "": "Thanks for reaching out — we'd like to help.",
}
_NEXT = {
    "billing_payment": "review the charge and sort out a refund if one is due",
    "account_access": "verify your details and get you back into your account",
    "trip_issue": "pull up the trip and look into what happened",
    "safety_incident": "escalate this to our safety team for an urgent review",
    "delivery_order": "look into your order and make it right",
    "service_complaint": "get this to a specialist and follow up properly",
    "general_query": "point you in the right direction",
    "": "look into this for you",
}


def _draft_rules(message: str, retrieved: list, intent: str = "") -> dict:
    """Offline synthesizer: build a NEW grounded reply from the top-3 references (no verbatim copy)."""
    ids = [r["pair_id"] for r in retrieved[:3]]
    dm_dominant = sum(any(mk in r["brand_reply"].lower() for mk in _DM_MARKERS)
                      for r in retrieved[:3]) >= 2
    ack = _ACK.get(intent, _ACK[""])
    nxt = _NEXT.get(intent, _NEXT[""])
    if intent == "safety_incident":
        draft = f"{ack} Please DM us the trip details now so we can {nxt} straight away."
        pattern = "move the customer to a private channel and escalate"
    elif dm_dominant:
        draft = f"{ack} If you DM us your account email and the trip details, we'll {nxt} as quickly as we can."
        pattern = "move to DM for account details, then resolve"
    else:
        draft = f"{ack} Reply here with your account email and the details and we'll {nxt}."
        pattern = "gather details, then follow up directly"
    return {
        "draft": draft,
        "cited_example_id": ids[0] if ids else "",
        "justification": (f"Synthesized from {len(ids)} similar historical cases "
                          f"({', '.join(ids)}); mirrors their resolution pattern "
                          f"({pattern}) without copying any reply verbatim."),
        "method": "rules-synth",
    }


def draft_reply(message: str, retrieved: list, intent: str = "") -> dict:
    """Grounded reply with a verification pass (Phase 12/13).
    Returns {"draft","cited_example_id","justification","method","verification"}.
    On LLM path: verify the draft; if it HARD-fails (invented URL / unsupported promise),
    regenerate once strictly; if it still fails, fall back to the safe reused-reply template."""
    if not retrieved:
        return {"draft": "", "cited_example_id": "", "justification": "no retrieval context",
                "method": "none", "verification": {"ok": False, "issues": ["no_retrieval"]}}
    if llm.llm_available():
        try:
            out = _draft_llm(message, retrieved)
            v = verify_draft(message, out["draft"], retrieved, intent)
            if not v["ok"]:
                out = _draft_llm(message, retrieved, strict=True)
                v = verify_draft(message, out["draft"], retrieved, intent)
            if not v["ok"]:                          # still bad -> safe grounded template
                out = _draft_rules(message, retrieved, intent)
                out["method"] += " (verify_fallback)"
                v = verify_draft(message, out["draft"], retrieved, intent)
            out["verification"] = v
            return out
        except Exception as e:
            out = _draft_rules(message, retrieved, intent)
            out["method"] = f"rules (llm_error: {type(e).__name__})"
            out["verification"] = verify_draft(message, out["draft"], retrieved, intent)
            return out
    out = _draft_rules(message, retrieved, intent)
    out["verification"] = verify_draft(message, out["draft"], retrieved, intent)
    return out


# =========================== ROUTING ===========================

def route_decision(classification: dict, message: str, thread_history: list | None = None) -> dict:
    """Multi-signal AI-vs-HUMAN routing with a plain-English reason (CLAUDE.md 6).

    Safety is decided by the CONTEXT-AWARE detect_safety() (LLM verifier or the NLP rule detector),
    not by blind substring matching. This fixes the old 'crash' collision: "the app keeps crashing"
    is understood as a software issue (AI) while "our car crashed" escalates (HUMAN). See the
    before/after in reports/failure_analysis.md."""
    thread_history = thread_history or []
    m = message.lower()
    reasons = []
    escalate = False

    conf = float(classification.get("confidence", 0.0))
    intent = classification.get("intent", "")

    safety = detect_safety(message)
    if safety["is_safety"]:
        escalate = True
        reasons.append(f"safety incident — {safety['reason']} [{safety['method']}]")
    elif intent == "safety_incident":
        # classifier says safety but the context check disagreed: escalate anyway (err on caution)
        # and flag the disagreement so it is auditable.
        escalate = True
        reasons.append(f"classifier intent=safety_incident (context check: {safety['reason']})")

    frustration = [w for w in _FRUSTRATION_WORDS if w in m]
    if len(frustration) >= 2:
        escalate = True
        reasons.append(f"high frustration ({len(frustration)} markers: {', '.join(frustration[:4])})")

    if conf < 0.55:
        escalate = True
        reasons.append(f"low classifier confidence ({conf:.2f} < 0.55)")

    # repeat-contact / escalation signal from thread reconstruction
    if len(thread_history) >= 3:
        escalate = True
        reasons.append(f"repeat contact ({len(thread_history)} prior messages in thread)")

    decision = "HUMAN" if escalate else "AI"
    if not escalate:
        reasons.append(f"confident ({conf:.2f}) non-safety intent '{intent}', no frustration/repeat signals")
    return {"decision": decision, "reason": f"Routed to {decision} because: " + "; ".join(reasons) + "."}


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    tests = [
        "why was I charged twice for one ride, I need a refund",
        "the app keeps crashing every time I open it",           # the intentional crash-collision bug
        "our driver was drunk and driving recklessly, I felt unsafe",
    ]
    for t in tests:
        c = classify_intent(t, mode="rules")
        r = retrieve_similar(t, k=2)
        d = draft_reply(t, r)  # rules path if no key
        route = route_decision(c, t, [])
        print("\nMSG:", t)
        print(" intent:", c)
        print(" top match:", r[0]["pair_id"], r[0]["score"])
        print(" route:", route["reason"])
