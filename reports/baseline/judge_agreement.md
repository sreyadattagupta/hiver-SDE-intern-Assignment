# LLM-Judge vs Human Agreement

Provider: **groq**  |  Drafts judged: **25** (deterministic rules-mode drafts)

| Dimension | % agreement | Cohen's kappa | human mean | judge mean |
|---|---|---|---|---|
| grounded | 48% | 0.085 | 0.84 | 0.40 |
| helpful | 60% | 0.219 | 0.52 | 0.20 |
| polite | 96% | 0.000 | 1.00 | 0.96 |

**Reading this honestly:** 'polite' has near-zero variance (Uber's templated replies are always courteous), so its kappa is meaningless — high agreement there is not evidence the judge is trustworthy. The dimensions that matter are *grounded* and especially *helpful*, where the judge must distinguish a real resolution from a polite 'please DM us' deflection. Trust the judge only as far as its kappa on *helpful* holds up.
