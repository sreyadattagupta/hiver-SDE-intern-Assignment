"""
Context-preserving text preprocessing for noisy Twitter support messages.

Design rule (critical): normalize noise WITHOUT collapsing meaning. In particular
"app keeps crashing" and "my driver crashed" must stay distinguishable — we never strip the
subject/entity words that carry safety context. See tests/test_preprocess.py.

Two functions:
  clean_for_embedding(text): light normalization used for retrieval/vectorization.
  normalize_repeats(text):   collapse elongated chars ("saaad" -> "saad") for robustness.
"""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|t\.co/\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")
_MULTISPACE = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{2,}")          # 3+ identical chars -> 1 ("crashingggg" -> "crashing")
_NONWORD_EDGES = re.compile(r"[^\w\s'#$/.,!?%-]")  # drop stray symbols, keep useful punctuation

# light, safe abbreviation expansion (Twitter support slang). Kept small + reversible in meaning.
_ABBREV = {
    r"\bu\b": "you", r"\bur\b": "your", r"\bpls\b": "please", r"\bplz\b": "please",
    r"\bthx\b": "thanks", r"\bacct\b": "account", r"\bappt\b": "appointment",
    r"\bmsg\b": "message", r"\bdm\b": "dm", r"\bcanceld\b": "cancelled",
    r"\brefnd\b": "refund", r"\bdriverz\b": "drivers",
}


def normalize_repeats(text: str) -> str:
    return _REPEAT.sub(r"\1", text)  # collapse 3+ elongations to one so noisy tokens match corpus


def clean_for_embedding(text: str, drop_handles: bool = True) -> str:
    """Normalize for vectorization while PRESERVING semantic subject words (driver/app/car/etc.)."""
    if not isinstance(text, str):
        return ""
    t = text
    t = _URL.sub(" URL ", t)
    if drop_handles:
        t = _HANDLE.sub(" ", t)          # @Uber_Support / @115873 carry no semantic content here
    t = normalize_repeats(t)
    low = t.lower()
    for pat, repl in _ABBREV.items():
        low = re.sub(pat, repl, low)
    low = _NONWORD_EDGES.sub(" ", low)
    low = _MULTISPACE.sub(" ", low).strip()
    return low


if __name__ == "__main__":
    for s in ["@Uber_Support the app keeps CRASHINGGGG!! https://t.co/x pls help",
              "my driver crashed the car @115873",
              "I was chargedddd twice, refnd plz"]:
        print(repr(s), "->", repr(clean_for_embedding(s)))
