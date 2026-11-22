# Experiment log

One row per scored run, newest at the bottom. Validation numbers are only comparable if they
use the shared split (see D-003 in `DECISIONS.md`); note the split version in "Val split".

| Date | Author | Script / commit | Val split | Val MAP@5 | Public LB | Submission file | Notes |
|---|---|---|---|---|---|---|---|
| — | — | `sample_submission.csv` | — | — | — | all `new_individual` | reference baseline, not yet submitted |
| 2026-11-22 | astral-fi via Claude | `notebooks/01_eda.ipynb` §3 | LOO on train, known ids only (not D-003) | hyena 0.878 / leopard 0.888 | — | — | Cosine k-NN, centred + L2 MiewID, identity score = max sim, no `new_individual`. Overstates test score: test has many new ids. |
| 2026-11-22 | astral-fi via Claude | `models/baseline_knn.py` | D-009 v1 (3 folds) | 0.9026 ±0.005 (hyena 0.8835, leopard 0.9097; known 0.8464, new 0.9742) | — | `submissions/baseline_knn.csv` | Cosine k-NN, centred+L2, max-sim per id; `new_individual` first if top-1 < 0.45 else #2. Test: new first for 50.6%. |
| 2026-11-22 | astral-fi via Claude | `models/newid_lr.py` | D-009 v1 | 0.9027 (hyena 0.8824, leopard 0.9103; known 0.8518, new 0.9676) | — | `submissions/newid_best.csv` | Gallery-trained LR P(new) + expected-AP placement, tau=0.02, bias −0.5. At new_frac 0.25: 0.8744 (baseline 0.8693); 0.55: 0.9281 (0.9310). |
| 2026-11-22 | astral-fi via Claude | `models/fusion_rerank.py` | D-009 v1 | 0.9073 (hyena 0.8854, leopard 0.9155; known 0.8578, new 0.9705) | — | `submissions/fusion_best.csv` | LR re-ranker on 16 features over top-20 ids; new score 0.75·(1−max prob). HGB 0.9062. |
| 2026-11-22 | astral-fi via Claude | `models/metric_learning.py` (cos+lda128_s0.3) | D-009 v1 | 0.9093 (hyena 0.8921, leopard 0.9157; known 0.8540, new 0.9799) | — | `submissions/metric_best.csv` | Mean of cosine and PCA256→LDA128 cosine; T hyena 0.465 / leopard 0.509 (tuned on same folds). Baseline re-tuned same way: 0.9036. |
| 2026-11-22 | astral-fi via Claude | `models/sift_rerank.py` | D-009 v1 | 0.9180 ±0.005 (hyena 0.8922, leopard 0.9277; known 0.8732, new 0.9752) | — | `submissions/sift_best.csv` | RootSIFT, mutual-NN+ratio 0.8, affine RANSAC 8px on top-10 ids × 2 imgs; score maxcos + 0.06·log1p(inl−3); T 0.475. ~5 min uncached. |
| 2026-11-22 | astral-fi via Claude | `models/combo_lda_sift.py` | D-009 v1 | **0.9235 ±0.004** (hyena 0.9009, leopard 0.9319; known 0.8848, new 0.9728) | — | `submissions/combo_lda_sift.csv` | **Current best (D-010).** cos+LDA blend + 0.09·log1p(inl−3); T hyena 0.495 / leopard 0.53 (tuned on same folds). Test: new first 45.4%. |
