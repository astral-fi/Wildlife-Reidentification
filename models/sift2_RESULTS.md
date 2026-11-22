# sift2: more SIFT coverage inside the cos+LDA combo (astral-fi via Claude, 2026-11-22)

Files: `models/sift2_feats.py` (candidates and pair stats, cache `cache/sift2_pairs.pkl`, val features `cache/sift2_feats.pkl`),
`models/sift2_tune.py` (grid of per-query arrays -> `cache/sift2_grid.pkl`), `models/sift2_report.py` (in-sample and honest tables),
`models/sift2_submit.py` (final config, `--eval` and `--submit`). Pair matching is `models.sift_match.pair_stats`, unchanged (RootSIFT,
mutual-NN, ratio 0.8, affine-partial RANSAC 8 px), and it uses only the provided keypoints.

## Setup
Per query: the union of the top-15 identities by maxcos and the top-15 by cos+LDA blend (about 16 ids). For each id, up to 5 best-cosine
gallery images are verified (about 81 pairs per query). Score: `blend + w_s*log1p(max(agg - c_s, 0))`. new_individual goes first iff top < T_s, else second.
Grid per species: candidate mode {cos, blend} x K {10, 15} x P {1, 2, 3, 5} x viewpoint-first {0, 1} x agg {max, sum of top-2} x
{raw, kp-normalised} x w x c x T x H (H = "never new if top inliers >= H", a 2-D new rule).
**Honest** = choose everything per species on 2 folds, score the 3rd, and rotate.

## Results (3 folds, MAP@5)
| config | in-sample | honest (2-tune/1-score) |
|---|---|---|
| combo_lda_sift (cos top-10 x 2 imgs, max, w .09, c 3, T per species) | 0.9235 | **0.9227 ±0.004** |
| same structure, w/c per species too | 0.9235 | 0.9215 |
| restricted to P <= 2 images/id | 0.9251 | 0.9230 |
| full grid without viewpoint | 0.9281 | 0.9257 |
| **full grid (final)** | **0.9282** | **0.9263 ±0.005** |
| full grid with 2-D (score, inliers) new rule | 0.9282 | 0.9261 |

The final config was chosen in-sample on all 3 folds (`--eval`: 0.9282, hyena 0.9021, leopard 0.9380, known 0.8912, new 0.9755, std 0.003):
- hyena: candidates = top-15 by blend, P=3, same/unknown-viewpoint gallery images first, max inliers, w=0.06, c=3, T=0.48
- leopard: candidates = top-15 by cosine, P=5, sum of top-2 inliers, w=0.09, c=6, T=0.55

Honest gain over the combo: +0.0036, about 1 fold-std. It shows up in every rotation. Almost all of it is leopard (0.932 -> 0.938) and known queries (0.885 -> 0.891).

## Runtime
- Val: about 218k pairs per fold (655k in total, 400k new beyond `sift_pairs.pkl`). With 6 workers this took 4.5-11 min per fold on a contended machine. Re-evaluation from cache takes 20 s per fold.
  The grid takes 3.5 min (6 procs) and the report takes 6.5 min.
- Test: 232k pairs (177k new), about 36 CPU-min, so **about 6-8 min with 6 free cores**. This run took 286 min wall because the machine was
  saturated or suspended. Submission: `submissions/sift2_best.csv`, with new_individual first for 45.0% of test images.

## Insights
- **More images per identity is the lever.** P <= 2 keeps only about 0.0003 of the honest gain. P=5 with sum-of-top-2 helps leopards, which have many gallery images per id.
  Two independent verified views beat one lucky match. The higher cutoff c=6 goes with the sum.
- Using blend instead of cosine to pick candidates, and K=15 instead of 10, matter little. Both are picked inconsistently across rotations.
- Viewpoint-first ordering of the gallery is a wash (+0.0004 honest). The best-cosine images already share the flank.
- SIFT for the new decision adds nothing: the H rule is picked once in 6 and gives -0.0002. Keypoint-count normalisation does not help either.
- Per-species w/c alone (combo structure) overfits slightly: honest 0.9215 vs 0.9227.
