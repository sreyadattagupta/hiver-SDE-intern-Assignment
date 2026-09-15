"""
Retrieval backends for grounding (Phase 4C — hybrid retrieval).

Methods (all runnable offline, no downloads):
  tfidf   — lexical: TF-IDF (1-2 grams) cosine. The original baseline.
  lsa     — semantic: TruncatedSVD (Latent Semantic Analysis) over the TF-IDF matrix, dense
            cosine. Captures synonymy/co-occurrence ("billed twice" ~ "charged two times") that
            pure lexical TF-IDF misses. This is genuine distributional semantics — it is NOT a
            transformer; we label it LSA everywhere, never "neural embeddings".
  hybrid  — fuse per-query min-max-normalised tfidf + lsa scores, then a light lexical reranker
            (token-Jaccard + exact-phrase bonus) reorders the top candidates.
  st      — sentence-transformers ('all-MiniLM-L6-v2'). Real neural embeddings, but the model
            weights download from the HuggingFace LFS CDN which is BLOCKED in this sandbox, so it
            is availability-gated: used only if importable AND the model loads locally, else the
            caller falls back. Never silently substituted — the method name reported reflects what
            actually ran.

All retrievers return: [{"pair_id","customer_msg","brand_reply","score","method"}].
`customer_msg`/`brand_reply` are the ORIGINAL text (for display); vectorisation uses the cleaned text.
"""
from __future__ import annotations

import functools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_for_embedding

PAIRS_PATH = "data/uber_pairs.csv"
LSA_COMPONENTS = 200


@functools.lru_cache(maxsize=1)
def _corpus():
    import os
    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"{PAIRS_PATH} missing — run `python -m src.build_pairs` first.")
    df = pd.read_csv(PAIRS_PATH)
    df["_clean"] = df["customer_msg"].fillna("").map(clean_for_embedding)
    return df.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _tfidf_index():
    df = _corpus()
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=20000)
    matrix = vec.fit_transform(df["_clean"])
    return vec, matrix


def tfidf_vectors(texts):
    """Transform arbitrary texts into the ALREADY-FITTED corpus TF-IDF space (no refit).
    Used by the verified-resolution retrieval layer so its scores share the historical space."""
    vec, _ = _tfidf_index()
    cleaned = [clean_for_embedding(t) for t in texts]
    return vec.transform(cleaned)


def tfidf_cosine(query: str, texts: list) -> np.ndarray:
    """Cosine similarity of `query` vs each text in the fitted TF-IDF space.
    Returns an empty array when `texts` is empty (deterministic, no exceptions)."""
    if not texts:
        return np.zeros(0)
    vec, _ = _tfidf_index()
    q = vec.transform([clean_for_embedding(query)])
    m = tfidf_vectors(texts)
    return cosine_similarity(q, m)[0]


@functools.lru_cache(maxsize=1)
def _lsa_index():
    vec, matrix = _tfidf_index()
    n_comp = min(LSA_COMPONENTS, matrix.shape[1] - 1, matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    dense = svd.fit_transform(matrix)          # (n_docs, n_comp)
    dense = normalize(dense)                    # unit vectors -> cosine == dot
    return svd, dense


def _rows(df, idxs, scores, method):
    out = []
    for i in idxs:
        r = df.iloc[i]
        out.append({"pair_id": str(r["pair_id"]), "customer_msg": str(r["customer_msg"]),
                    "brand_reply": str(r["brand_reply"]), "score": round(float(scores[i]), 4),
                    "method": method})
    return out


def _tfidf_scores(query: str):
    vec, matrix = _tfidf_index()
    q = vec.transform([clean_for_embedding(query)])
    return cosine_similarity(q, matrix)[0]


def _lsa_scores(query: str):
    vec, matrix = _tfidf_index()
    svd, dense = _lsa_index()
    q = normalize(svd.transform(vec.transform([clean_for_embedding(query)])))
    return (dense @ q[0])


def _minmax(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _lexical_rerank(query: str, cand_idx, base_scores, top_k):
    """Reorder candidates by token-Jaccard + exact-phrase bonus (dependency-free cross-check)."""
    df = _corpus()
    qtok = set(clean_for_embedding(query).split())
    reranked = []
    for i in cand_idx:
        dtok = set(str(df.iloc[i]["_clean"]).split())
        jac = len(qtok & dtok) / len(qtok | dtok) if (qtok | dtok) else 0.0
        # blend: keep fusion score dominant, nudge by lexical precision
        reranked.append((i, 0.7 * float(base_scores[i]) + 0.3 * jac))
    reranked.sort(key=lambda t: t[1], reverse=True)
    return [i for i, _ in reranked[:top_k]]


def retrieve(query: str, k: int = 3, method: str = "auto") -> list:
    df = _corpus()
    if method == "auto":
        # evidence (reports/retrieval_eval.md): neural 'st' wins intent-match@3 (0.715 vs 0.610);
        # use it when the model is available, else fall back to the download-free hybrid.
        if st_available():
            try:
                return retrieve(query, k, "st")
            except Exception:
                pass
        return retrieve(query, k, "hybrid")
    if method == "tfidf":
        s = _tfidf_scores(query)
        idx = s.argsort()[::-1][:k]
        return _rows(df, idx, s, "tfidf")
    if method == "lsa":
        s = _lsa_scores(query)
        idx = s.argsort()[::-1][:k]
        return _rows(df, idx, s, "lsa")
    if method == "st":
        model = _st_model()  # raises if unavailable — caller decides fallback
        s = _st_scores(query, model)
        idx = s.argsort()[::-1][:k]
        return _rows(df, idx, s, "st")
    # hybrid (default): fuse normalised tfidf+lsa, then lexical rerank top candidates
    st, sl = _minmax(_tfidf_scores(query)), _minmax(_lsa_scores(query))
    fused = 0.5 * st + 0.5 * sl
    cand = fused.argsort()[::-1][:max(k * 4, 12)]
    idx = _lexical_rerank(query, cand, fused, k)
    return _rows(df, idx, fused, "hybrid+rerank")


# ---------- sentence-transformers (gated; needs HuggingFace weights = local-only here) ----------
@functools.lru_cache(maxsize=1)
def st_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        _st_model()
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=1)
def _st_embeddings():
    df = _corpus()
    model = _st_model()
    emb = model.encode(df["_clean"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _st_scores(query: str, model):
    q = model.encode([clean_for_embedding(query)], normalize_embeddings=True)[0]
    return _st_embeddings() @ q


if __name__ == "__main__":
    q = "Uber billed my card two times for the same trip"
    for meth in ("tfidf", "lsa", "hybrid"):
        top = retrieve(q, k=3, method=meth)
        print(f"\n[{meth}]")
        for r in top:
            print(f"  {r['score']:.3f} {r['pair_id']}  {r['customer_msg'][:70]}")
    print("\nsentence-transformers available here:", st_available())
