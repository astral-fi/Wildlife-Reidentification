# Transductive / graph methods on top of combo_lda_sift (agent: trans)

Code: `models/trans_graph.py` (run `OMP_NUM_THREADS=4 python -m models.trans_graph [base kr cl pl]`,
`--submit` for test). Uses the combo's cos+LDA blend and SIFT bonus (w=0.09, c=3); reads
`cache/sift_pairs.pkl` read-only (uncached pairs at other new-rates are matched in memory, never saved).

**Honest** = parameters + per-species T chosen on 2 folds, scored on the 3rd (rotated).
In-sample = tuned on all 3 folds (as D-010 reports).

| Method (grid) | in-sample best | honest |
|---|---|---|
| combo_lda_sift (T only) | 0.9236 | **0.9224 ±0.0042** |
| 1. k-reciprocal, lam∈{.1,.2,.3}, (k1,k2)∈{(10,3),(20,6)} | 0.9255 (lam .2, k1 10, k2 3) | **0.9242 ±0.0047** |
| 1b. k-recip finer grid (k1 5–10, lam .15–.3) | 0.9255 | 0.9239 ±0.0051 |
| 2. query clustering (mutual-3NN, tau .5–.7, beta 0–.4, pooled new decision) | 0.9239 | 0.9225 ±0.0039 |
| 3. pseudo-labelling (Tc .6–.8, gamma .9/1) | 0.9241 | 0.9224 ±0.0037 |
| k-recip + clustering/pseudo-label | 0.9258 | 0.9249 ±0.0050 (picks differ by fold) |

Per fold, k-recip (lam .2,k1 10,k2 3) vs combo, honest: +0.0027 / +0.0017 / +0.0010.

Robustness (same fixed config, honest = T tuned on 2 folds):

| new_frac | combo | k-recip |
|---|---|---|
| 0.25 | 0.9011 | 0.9012 |
| ~0.44 (default) | 0.9224 | 0.9242 |
| 0.55 | 0.9484 | 0.9504 |

Chosen: `BEST = dict(lam=0.2, k1=10, k2=3)`, T retuned on all 3 folds (hyena 0.45, leopard 0.545; in-sample 0.9255) → `submissions/trans_best.csv` (new_individual first in 44.9% of test; all test SIFT pairs were cached).

Insights
- The gain from k-reciprocal grows with the new-rate (0 at 25%, +0.002 at 44–55%): it mainly
  sharpens the new decision (Jaccard pulls queries whose neighbourhood is other queries away
  from gallery ids). Test is likely high-new, so it should help there.
- Small k1 (≈10) works; Zhong's default k1=20 hurts at lam≥0.2 — many ids have 1–3 gallery images.
- Query clustering and one-round pseudo-labelling give ≈0 honest: the base score already calls
  new queries correctly ~97%, and propagating scores inside clusters mixes in errors as often as fixes.
- Stacking cluster/pseudo-label on k-recip adds +0.0007 honest but with unstable picks — not adopted.
- Overall the transductive lever is small (+0.002); remaining loss is known-id ranking (~0.885).
