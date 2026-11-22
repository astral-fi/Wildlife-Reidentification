# Metric learning / re-ranking results (models/metric_learning.py)

Shared 3-fold split, mean MAP@5. Every variant: per-species embedding/scores, new_individual
first if top-1 identity score < T[species] else at #2; T swept per species on the 3 folds
(note: T tuned on the same folds, so all rows are slightly optimistic, equally).
Run: `python -m models.metric_learning [variant ...]`.

| variant | all | hyena | leopard | known | new |
|---|---|---|---|---|---|
| base (cos, max, re-tuned T) | 0.9036 | 0.8843 | 0.9108 | 0.8541 | 0.9668 |
| LDA64 only (PCA256, eigen, shrink 0.1) | 0.8945 | 0.8774 | 0.9009 | 0.8373 | 0.9675 |
| LDA128 only, shrink 0.1 | 0.8992 | 0.8879 | 0.9034 | 0.8461 | 0.9669 |
| LDA128 only, shrink 0.3 | 0.9022 | 0.8885 | 0.9073 | 0.8502 | 0.9685 |
| cos + LDA128 s0.1 (mean of sims) | 0.9083 | 0.8940 | 0.9137 | 0.8563 | 0.9747 |
| **cos + LDA128 s0.3** | **0.9093** | 0.8921 | 0.9157 | 0.8540 | 0.9799 |
| cos + LDA128 s0.5 | 0.9086 | 0.8910 | 0.9152 | 0.8520 | 0.9809 |
| cos + LDA200 s0.3 | 0.9095 | 0.8929 | 0.9157 | 0.8531 | 0.9814 |
| cos + 0.5*LDA128 s0.3 | 0.9083 | 0.8904 | 0.9150 | 0.8533 | 0.9786 |
| cos + 2*LDA128 s0.3 | 0.9080 | 0.8922 | 0.9140 | 0.8524 | 0.9790 |
| PCA-whiten 256 only | 0.8961 | 0.8878 | 0.8992 | 0.8446 | 0.9618 |
| cos + whiten 256 | 0.9049 | 0.8922 | 0.9097 | 0.8583 | 0.9644 |
| agg 0.7*max + 0.3*2nd-best | 0.9034 | 0.8808 | 0.9119 | 0.8518 | 0.9693 |
| alpha-QE k=3 (a=3) | 0.9023 | 0.8843 | 0.9090 | 0.8494 | 0.9697 |
| alpha-QE k=5 | 0.9004 | 0.8816 | 0.9074 | 0.8469 | 0.9686 |
| viewpoint +0.05 same vp | 0.9025 | 0.8802 | 0.9108 | 0.8476 | 0.9725 |
| viewpoint +0.05 same, -0.05 flipped | 0.9023 | 0.8799 | 0.9107 | 0.8473 | 0.9725 |
| cos+LDA s0.3 + vp 0.05 | 0.9070 | 0.8873 | 0.9144 | 0.8533 | 0.9755 |

Best: cos + LDA (PCA 256 on centred+L2 gallery -> LDA eigen shrinkage 0.3, 128 dims, re-centred
+L2), similarity = mean of the two cosines, max per identity, T = hyena 0.465 / leopard 0.509.
+0.006 over re-tuned baseline (~0.9 std-free; fold std not measured — treat as small but
consistent across variants). Submission: submissions/metric_best.csv.

Didn't help: LDA or whitening alone, QE, top-2 aggregation, viewpoint bonus/flip penalty.
Not tried (time): NCA, k-reciprocal re-ranking, DBA.
