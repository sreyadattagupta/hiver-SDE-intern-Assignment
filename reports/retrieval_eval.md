# Retrieval comparison (proxy metric)

Golden queries: 200 · k=3 · metric = intent-match@3 (top-k contains a pair whose rule-intent == query's true intent).

sentence-transformers available in this env: **True**

| method | intent-match@3 | mean top-1 sim | ms/query |
|---|---|---|---|
| tfidf | 0.610 | 0.423 | 1.9 |
| lsa | 0.605 | 0.617 | 8.8 |
| hybrid | 0.615 | 0.941 | 4.5 |
| st ⭐ | 0.715 | 0.625 | 38.3 |

**Winner by proxy: `st`.** Note: `st` (neural) similarity is on a different scale than tfidf/lsa cosine, so compare methods by intent-match, not raw similarity.

_Proxy caveat: intent-match uses the rule classifier to label retrieved pairs, so it measures topical alignment, not human-judged usefulness._
