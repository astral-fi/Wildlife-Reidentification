# Fusion / learned re-ranker results (models/fusion_rerank.py)

Shared 3-fold split, mean MAP@5. Re-ranker trained only on pseudo-queries simulated from the
gallery (2 reps; held-out ids <=20 imgs -> new, 25% of kept ids' images -> known), per species.
Top-20 candidate ids, 16 features (max/mean/2nd cosine, rank, gap to best, log count,
same-viewpoint max sim + fraction, opposite-flank max sim + fraction, query top1/top2/gap,
image-level top1/top3, missing-viewpoint flag). new score = alpha * (1 - sum|max p).

| Variant | all | hyena | leopard | known | new |
|---|---|---|---|---|---|
| baseline_knn (T=0.45) | 0.9026 | 0.8835 | 0.9097 | 0.8464 | 0.9742 |
| **LR, max, alpha=0.75** | **0.9073** | 0.8854 | 0.9155 | 0.8578 | 0.9705 |
| LR, max, alpha=1.0 | 0.9072 | 0.8847 | 0.9156 | 0.8521 | 0.9775 |
| LR, sum, alpha=1.0 | 0.9043 | 0.8841 | 0.9118 | 0.8584 | 0.9628 |
| HGB, max, alpha=1.5 | 0.9062 | 0.8820 | 0.9153 | 0.8541 | 0.9728 |
| HGB, sum, alpha=3.0 | 0.9000 | 0.8773 | 0.9086 | 0.8539 | 0.9589 |

Fold std ~0.004, so gain over baseline (+0.005) is ~1 std.
Submission: submissions/fusion_best.csv (LR, max, alpha=0.75).

What didn't help: HGB is no better than logistic regression (plateaus; features are near-linear);
P(new)=1-sum p is worse than 1-max p. Not done (time): per-identity PCA+logistic/SVM
classifiers and MLP re-ranker.
