# new_individual decision — results (models/newid_lr.py)

Known-id ranking = baseline (max cosine per identity, centred+L2 MiewID). Only the new
decision/placement changes. P(new) = per-species logistic regression on
[top1, top1-top2, top2-top5, log gallery size of best id, query flank matches best id's flanks,
flank opposite, top1*opposite], trained inside predict by simulating on the gallery
(leave-one-out → known, leave-identity-out → new). List = candidates sorted by probability
(new: P(new); id: (1-P(new))·softmax(s/tau)) = expected-AP-optimal ordering. `bias` shifts the
logit of P(new).

| variant (3-fold, new_frac 0.40) | all | hyena | leopard | known | new |
|---|---|---|---|---|---|
| baseline (T=0.45, new at #2) | 0.9026 | 0.8835 | 0.9097 | 0.8464 | 0.9742 |
| LR tau=0.05 | 0.8995 | 0.8840 | 0.9053 | 0.8305 | 0.9875 |
| LR tau=0.02 | 0.9004 | 0.8840 | 0.9066 | 0.8364 | 0.9821 |
| LR tau=0.10 | 0.8942 | 0.8769 | 0.9007 | 0.8185 | 0.9908 |
| LR tau=0.05 + query-graph P(new) smoothing (sim>0.7) | 0.8999 | 0.8839 | 0.9058 | 0.8311 | 0.9876 |
| LR tau=0.05 + smoothing (sim>0.8) | 0.8996 | 0.8839 | 0.9054 | 0.8306 | 0.9875 |
| **LR tau=0.02 bias=-0.5** (best) | **0.9027** | 0.8824 | 0.9103 | 0.8518 | 0.9676 |
| LR tau=0.01 bias=-0.5 | 0.9022 | 0.8805 | 0.9103 | 0.8536 | 0.9642 |
| LR tau=0.03 bias=-1.0 | 0.9020 | 0.8782 | 0.9108 | 0.8616 | 0.9535 |
| LR tau=0.02 bias=-1.0 | 0.8989 | 0.8733 | 0.9084 | 0.8639 | 0.9434 |
| LR tau=0.02 bias=-1.5 | 0.8865 | 0.8488 | 0.9006 | 0.8770 | 0.8987 |

Robustness to new-rate (all MAP@5):

| method | new_frac 0.25 | 0.40 | 0.55 |
|---|---|---|---|
| baseline | 0.8693 | 0.9026 | 0.9310 |
| LR tau=0.02 bias=0 | 0.8656 | 0.9004 | 0.9293 |
| LR tau=0.02 bias=-0.5 | 0.8744 | 0.9027 | 0.9281 |

Submission: `submissions/newid_best.csv` (LR tau=0.02 bias=-0.5).

What didn't help / notes:
- Gains over the simple threshold+hedge baseline are within noise (±0.005). The new
  decision is already near its ceiling: new queries score 0.97–0.99; the loss is in known
  ranking (known ≈ 0.85), which this topic deliberately left fixed.
- Gallery-simulated LR over-predicts new (simulated new/known ratio ≈ 1:1 and LOO "known"
  queries are easier than real ones); a negative logit bias is needed.
- Transductive smoothing of P(new) over the query similarity graph changed nothing.
- Not tried in the time box: GMM/KDE open-set features, cluster-level propagation of matches.
