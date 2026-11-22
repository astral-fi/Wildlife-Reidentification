# fusion2 — learned LR re-ranker with LDA-blend + SIFT features (2026-11-22, astral-fi via Claude)

Files: `models/fusion2_rerank.py` (model + honest/in-sample eval), `models/fusion2_combo_honest.py`
(same protocol for `combo_lda_sift`). Pair cache: `cache/fusion2_pairs.pkl`. It is seeded read-only
from `cache/sift_pairs.pkl`, and `sift_match.py` is not modified.

Honest = tune alpha/beta (or w, T) per species on 2 folds, score the 3rd, rotate. In-sample = tune on all 3.

| model | protocol | all | hyena | leopard | known | new |
|---|---|---|---|---|---|---|
| combo_lda_sift | in-sample | 0.9235 ±0.0037 | 0.9009 | 0.9319 | 0.8848 | 0.9728 |
| combo_lda_sift | honest | **0.9221** ±0.0046 | 0.8991 | 0.9307 | 0.8808 | 0.9748 |
| fusion2, new = alpha·(1−max p) | in-sample | 0.9205 ±0.0048 | 0.8997 | 0.9282 | 0.8779 | 0.9748 |
| fusion2, new = alpha·(1−max p) | honest | 0.9195 ±0.0039 | 0.8970 | 0.9280 | 0.8772 | 0.9735 |
| fusion2, separate P(new) LR, new = beta·pnew | in-sample | 0.9223 ±0.0061 | 0.9018 | 0.9300 | 0.8808 | 0.9754 |
| fusion2, separate P(new) LR | honest | 0.9216 ±0.0065 | 0.9008 | 0.9293 | 0.8809 | 0.9735 |

Verdict: fusion2 does **not** beat the combo honestly (0.9216 vs 0.9221, which is within noise). No submission was written.

Config: candidates = top-10 cosine ids × 2 best-cosine images. 20 features per candidate: cos max/mean/rank/gap,
LDA-blend score/gap, same/opposite-viewpoint max cos, log count, log1p aff inliers (max, second),
max aff/sqrt(min kp), log1p ratio matches, log1p query kp, kp≥50 flag, top1 blend, top1−top2 blend,
log1p max aff over all candidates, relative aff, missing viewpoint. The LR (C=1, standardised) is trained on 2 reps of
pseudo-queries simulated from the gallery only (about 40% new). Each rep gets its own LDA and its own SIFT pairs.
The P(new) LR uses 8 query-level features. Runtime: about 12.5 min for 3 folds with a cold cache (fold 0 took 6 min
of SIFT matching; later folds reuse cached pairs). The combo honest check took about 3 min.

Insights:
1. Honest tuning costs the combo only 0.0014, so its T and w are not badly overfit. Hyena T moves between 0.48 and 0.505 across folds.
2. The learned LR does not improve known-id ranking over the hand-tuned `blend + 0.09·log1p(aff−3)`. That
   1-parameter additive form is close to what a linear model can extract from these features.
3. A separate P(new) model beats alpha·(1−max p) by about 0.002. Re-ranker probabilities are poorly calibrated
   for the "none of these" case.
4. The pseudo-queries come from a smaller gallery, so their score distribution is shifted. This train/test mismatch
   probably limits the learned model. Non-linear models (HGB) or pairwise interactions would be the next things to try.
