# Experiment log

One row per scored run, newest at the bottom. Validation numbers are only comparable if they
use the shared split (see D-003 in `DECISIONS.md`); note the split version in "Val split".

| Date | Author | Script / commit | Val split | Val MAP@5 | Public LB | Submission file | Notes |
|---|---|---|---|---|---|---|---|
| — | — | `sample_submission.csv` | — | — | — | all `new_individual` | reference baseline, not yet submitted |
| 2026-11-22 | astral-fi via Claude | `notebooks/01_eda.ipynb` §3 | LOO on train, known ids only (not D-003) | hyena 0.878 / leopard 0.888 | — | — | Cosine k-NN, centred + L2 MiewID, identity score = max sim, no `new_individual`. Overstates test score: test has many new ids. |
| 2026-11-22 | astral-fi via Claude | `models/baseline_knn.py` | D-009 v1 (3 folds) | 0.9026 ±0.005 (hyena 0.8835, leopard 0.9097; known 0.8464, new 0.9742) | — | `submissions/baseline_knn.csv` | Cosine k-NN, centred+L2, max-sim per id; `new_individual` first if top-1 < 0.45 else #2. Test: new first for 50.6%. |
