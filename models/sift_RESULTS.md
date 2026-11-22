# SIFT verification and re-ranking: results (astral-fi via Claude, 2026-11-22)

Files: `models/sift_match.py` (pair matching with cache in `cache/sift_pairs.pkl`), `models/sift_rerank.py`
(`--analyse` runs the AUC tables and grid; `--eval` and `--submit` use `--cfg=dict(...)`).
Only the provided keypoints and descriptors are used. OpenCV does matching and geometric verification only.

## Matching pipeline
RootSIFT (L1 norm then sqrt), at most 1500 keypoints per image (deterministic subsample). Brute-force L2 2-NN,
**mutual-NN + ratio < 0.8**, then RANSAC on the matches. Pairs where either image has fewer than 2 keypoints score 0 inliers,
which means "no evidence" (the score falls back to cosine).

## Hard-pair AUC (known queries vs gallery images of their top-10 MiewID identities, 3 folds)
| species | ratio0.8 | ratio0.7 | affine-partial 8px | fundamental MAGSAC 3px | homography 20px | ratio/sqrt(min n) | MiewID cos |
|---|---|---|---|---|---|---|---|
| hyena (n=24.6k, 9% pos) | 0.733 | 0.765 | 0.735 | 0.728 | 0.737 | 0.745 | 0.945 |
| leopard (n=65.2k, 10% pos) | 0.875 | 0.879 | 0.876 | 0.850 | 0.877 | 0.879 | 0.954 |

New-vs-known AUC from top-1 evidence: hyena cos1 0.901, max affine inliers 0.749. Leopard cos1 0.941, max affine inliers 0.880.

## Re-ranking (score = maxcos + w*log1p(max(inliers - c, 0)), top 10 ids x 2 best images each; new first if top score < T, else new at #2)
| config | all | hyena | leopard | known | new |
|---|---|---|---|---|---|
| baseline (w=0, T=0.45) | 0.9026 | 0.8835 | 0.9097 | 0.8464 | 0.9742 |
| **affine, w=0.06, c=3, T=0.475 (best, submitted)** | **0.9180** | 0.8922 | 0.9277 | 0.8732 | 0.9752 |
| homography 20px, w=0.06, c=5, T=0.475 | 0.9183 | 0.8925 | 0.9279 | 0.8733 | 0.9756 |
| fundamental, w=0.06, c=8, T=0.475 | 0.9166 | 0.8898 | 0.9266 | 0.8687 | 0.9778 |
| ratio-only count, w=0.04, c=8, T=0.45 | 0.9155 | 0.8931 | 0.9239 | 0.8765 | 0.9654 |

Fold std is 0.005. The affine and homography results are tied, and I chose affine because it is the more defensible model with fewer degrees of freedom.

## Runtime
About 8 ms per pair. Validation needed about 128k pairs, which took 5 min with 6 workers (cached after that, so evaluate takes 3 s per fold).
Test needed 55k pairs (about 2 min). Submission: `submissions/sift_best.csv` (new_individual first for 47.0% of test images).

## What didn't help / notes
- The geometric model hardly matters once mutual-NN + ratio filtering is on. Affine, homography (large threshold) and raw match count are all within 0.003 AUC.
  The fundamental matrix is slightly worse. The EDA's weak AUC came from random pairs and from a plain 10px homography without mutual-NN.
- Normalising by keypoint count adds ~0.01 AUC on hyena only. I did not use it in the grid.
- Used alone, SIFT is much weaker than MiewID, but it is complementary: almost all the gain is on known queries (+0.027).
  Hyena (blurry spots) gains less than leopard.
- Using SIFT to decide new_individual adds little beyond letting T rise a bit. max-inlier new-vs-known AUC is below cos1.
- Not tried: per-species w/T, viewpoint-aware verification, more than 2 images per identity, ratio 0.7 inside the score.
