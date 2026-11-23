# Wildlife Re-Identification — shared context for all coding agents

This is the single source of truth for every AI agent working on this repo (Claude Code,
Codex, Cursor, Copilot, Gemini, Aider, or anything else) and for the humans. Several people
work here, each with their own agent. `CLAUDE.md`, `GEMINI.md` and
`.github/copilot-instructions.md` only point to this file — **edit this file, never those.**

## Before you start

1. `git pull`.
2. Read this file, then `docs/DECISIONS.md` and `docs/EXPERIMENTS.md`. They are the team's
   shared memory; don't redo or contradict a logged decision without superseding it.

## Your duties while working (every agent, every session)

1. **Log every decision** in `docs/DECISIONS.md`: append a new entry (next `D-xxx` id),
   never rewrite old ones. To reverse one, add a new entry that says `Supersedes: D-xxx` and
   change only the old entry's status line. A "decision" is anything another agent would
   otherwise re-derive or re-litigate: validation scheme, preprocessing, model choice,
   thresholds, file layout, conventions.
2. **Log every scored run** in `docs/EXPERIMENTS.md`: date, author, script/commit, val split,
   validation MAP@5, public LB if submitted, submission file, notes.
3. **Keep this file current.** When a fact here goes stale (data layout, environment, scripts
   and how to run them, best current approach), fix it in this file in the same commit as the
   change that made it stale. Add a short "Code map" section once there is code.
4. **Stay inside the allowed models and libraries** (see "Rules" below). Breaking them is
   academic misconduct. If unsure whether something is allowed, don't use it — add it to
   "Open questions" in `docs/DECISIONS.md` for the humans to ask the TA.
5. **Never commit the data** (`projectwild3/` is git-ignored) or large generated files
   (`*.npy`, `*.npz`, caches). Submissions go in `submissions/` (also ignored); record which
   file was submitted in `docs/EXPERIMENTS.md`.
6. Commit the docs updates together with the work they describe, keep commits small, and
   sign the author field in the logs as `<person> via <agent>` (e.g. `astral-fi via Claude`).

## Problem (Problem Statement 3)

Recognise individual spotted **hyenas** and **leopards** from precomputed image features, and
say when an animal has never been seen before. For each of the 2,791 test images, submit up to
5 guesses, best first. Each guess is a training `identity` of the same species, or
`new_individual`.

### Data (in `projectwild3/`, not in git — get `projectwild3.zip` from the course page)

Unzip `projectwild3.zip` at the repo root. Layout (paths relative to the repo root):

| Path | Contents |
|---|---|
| `projectwild3/reid_student_data_main/train.csv` | 7,163 rows: `id, species, viewpoint, identity` |
| `projectwild3/reid_student_data_main/test.csv` | 2,791 rows: `id, species, viewpoint` |
| `projectwild3/reid_student_data_main/miewid_features.npy` | 9,954 × 2,152 global MiewID re-id descriptors (float32), train rows then test rows |
| `projectwild3/reid_student_data_main/miewid_ids.csv` | 9,954 rows: `id` of each descriptor row, same order |
| `projectwild3/reid_student_data_main/sample_submission.csv` | 2,791 rows: `id,predictions` (all `new_individual`) |
| `projectwild3/sift_features_train.npz/` | **a directory** (the zip extracted the .npz): `<id>_xy.npy` (n×2 float32 px) and `<id>_desc.npy` (n×128 uint8) per train image, 7,163 × 2 files |
| `projectwild3/sift_features_test.npz/` | same for the 2,791 test images |
| `projectwild3/train_images_hyena/train_images/*.jpg`, `projectwild3/train_images_leopard/train_images/*.jpg` | the training cut-out images (1,912 + 5,251). **Do not use these — see below.** |

Because the `.npz` files arrived extracted, load SIFT per image with
`np.load(f"projectwild3/sift_features_train.npz/{id}_xy.npy")` (and `_desc.npy`), not
`np.load("....npz")[key]` as the statement shows. If you have the original `.npz` file
instead, the statement's way works; code should handle both.

**The training images must not feed any model or feature.** The rules forbid computing new
features from images and say the provided features are the only allowed representation of the
images. Whether even looking at them (e.g. to sanity-check matches) is OK is an open question
for the TA — until answered, don't open them from code at all.

Facts from the statement:
- Ids (`tr_*`, `te_*`) carry no information.
- A test image always belongs to the species shown — only guess identities of that species.
- Some test individuals are **not in train** → correct answer is `new_individual`. The same new
  individual can appear in several test images.
- Many individuals have only 1–2 training images; some have dozens.
- SIFT keypoints are in pixels of the cut-out (longer side ≤ 512 px). ~1/5 of images have
  <500 keypoints, ~5% have <50, a few have 0 (arrays with 0 rows).
- `viewpoint` ∈ {left, right, front, back, frontleft, frontright, backleft, backright, up, down};
  sometimes empty.
- There is no separate validation file — we must build our own split.

Facts we measured (2026-11-21):
- Train species: leopard 5,251 / hyena 1,912. Test: leopard 1,574 / hyena 1,217.
  Hyenas are a much larger share of test than train (44% vs 27%) — possibly many new hyenas.
- Hyena: 205 identities, median 6 imgs/id, 3 singletons, max 73.
  Leopard: 345 identities, median 3 imgs/id, 99 singletons, 150 with ≤2 imgs, max 391.
- Viewpoint dominated by left/right (~69% train, ~79% test); 136 train / 42 test missing.
- MiewID descriptor rows are **not** L2-normalised (norms ~60–80).

### Submission format

CSV with header, exactly one row per test id (2,791 rows):
```
id,predictions
te_00001,LEO0317 new_individual LEO0102
te_00002,new_individual
```
1–5 space-separated labels, best first, no repeats. Unknown labels just never score.
Missing / duplicate / unknown ids are rejected.

### Metric: MAP@5

Each image scores `1/k` if the correct label is the k-th guess (k ≤ 5), else 0; average over
images. Only the first correct guess counts, so where to put `new_individual` in the ranked
list matters as much as the ranking of known ids.
Public LB = random 30% of test; final ranking = other 70% (don't overfit the public LB).

## Rules (enforced — breaking them is academic misconduct)

**Inputs:** only the provided features. They may be transformed, combined, reduced, and matched
in any way the allowed models permit.

**Allowed models** (as in the course / Bishop, PRML):
- Linear, polynomial, ridge, lasso, Bayesian linear regression; logistic & softmax regression; GLMs.
- Discriminant analysis (incl. LDA as a projection), naive Bayes, k-NN and distance-based
  retrieval, kernel density estimation.
- Kernel methods, Gaussian processes, SVMs, metric learning with **linear** projections.
- Mixture models & EM, k-means, hierarchical and graph clustering, PCA and other linear
  dimensionality reduction.
- Decision trees, random forests, bagging, stacking, simple model combination, gradient
  boosting (XGBoost, LightGBM, CatBoost, sklearn `HistGradientBoosting`).
- MLPs trained from scratch on the provided features (plain fully-connected, e.g. sklearn
  `MLPClassifier`).
- Classical keypoint matching on the provided SIFT descriptors: NN matching, ratio test,
  RANSAC and other geometric verification.

**Allowed libraries:** numpy, pandas, scipy, scikit-learn, statsmodels, imbalanced-learn,
xgboost, lightgbm, catboost, networkx, opencv-python (matching & geometric verification of the
provided keypoints only), faiss-cpu, matplotlib, seaborn. Ask the TA before anything else.

**Not allowed:**
- Any data other than the competition files (no finding/downloading original images or labels).
- Computing new features from images; any pre-trained model.
- Deep-learning frameworks (PyTorch, TensorFlow, Keras, JAX); conv / recurrent / attention layers.
- AutoML tools (auto-sklearn, TPOT, H2O AutoML, …).

## Code map

| Path | What it is |
|---|---|
| `reid/data.py` | `load()` → train, test, X_train, X_test (aligned); `load_sift(id)`; `preprocess` (D-007). |
| `reid/split.py` | Shared validation split v2 (D-009 + D-013 cross-view), `make_split(seed, cross_frac=0.5)`, `cross_view_mask`, `describe(seed)`. |
| `reid/evaluate.py` | `evaluate(predict)` (3-fold MAP@5) and `make_submission(predict, name)`; documents the `predict` interface every model implements. |
| `reid/metric.py` | `map5`, `check_submission`. |
| `models/baseline_knn.py` | Baseline: `python -m models.baseline_knn [--submit]` (val 0.9026). |
| `models/combo_lda_sift.py` | Previous best (D-010), v1 val 0.9235 (v2 0.8119). `python -m models.combo_lda_sift [--submit]` (fast once `cache/sift_pairs.pkl` exists; otherwise ~5 min val + ~2 min test SIFT matching). |
| `models/metric_learning.py` | PCA/LDA/whitening/QE/viewpoint variants; `python -m models.metric_learning [variant ...]`. Results: `models/metric_RESULTS.md`. |
| `models/sift_match.py`, `models/sift_rerank.py` | RootSIFT matching + affine RANSAC with disk cache and multiprocessing; SIFT re-ranking. Results: `models/sift_RESULTS.md`. |
| `models/sift2_*.py` | Wider SIFT coverage, per-species (honest 0.9263 on D-009 v1). Results: `models/sift2_RESULTS.md`. |
| `models/trans_graph.py` | k-reciprocal re-ranking, query clustering, pseudo-labels. Results: `models/trans_RESULTS.md`. |
| `models/fusion2_*.py` | LR re-ranker with SIFT+LDA features; honest-tuning harness for the combo. Results: `models/fusion2_RESULTS.md`. |
| `models/sideaware.py` | v2 honest 0.8553 but worse on LB (D-014). Side bonus for identities lacking the query's flank. `python -m models.sideaware [--submit]` (features cached in `cache/sideaware_feats.pkl`). |
| `models/b1_pipeline.py` | b1's notebook (LB 0.83484) as an importable module; `python -m models.b1_pipeline` writes `submissions/b1/*.csv` (~13 CPU-min). Needs `lightgbm`. |
| `models/b1_eval.py` | Runs b1's pipeline on our v2 folds → `cache/b1_folds<B1_TAG>.pkl` (same env switches as the pipeline). |
| `models/b1_compare.py` | Compare cached b1 variants on v2: `python -m models.b1_compare "" _refs4`. |
| `models/ens_b1.py` | Weighted reciprocal-rank fusion of b1 with our models (no gain, D-015). |
| `models/stress_crossview.py` | Cross-view known-animal stress check (D-012). |
| `models/fusion_rerank.py` | Logistic/HGB candidate re-ranker on cosine+viewpoint features. Results: `models/fusion_RESULTS.md`. |
| `models/newid_lr.py` | Calibrated P(new) + expected-AP placement. Results: `models/newid_RESULTS.md`. |
| `notebooks/01_eda.ipynb` | EDA: labels, MiewID similarity, LOO retrieval baseline, test-vs-train "knownness", SIFT counts and matching, and a ranked list of models to try (§7). Committed with outputs; re-run with `jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb` (~1.5 min). |

## Key EDA findings (details in `notebooks/01_eda.ipynb`)

- **Test is far less "known" than train**: median nearest-train cosine 0.44 / 0.46 for test vs
  0.60 / 0.70 for train-LOO known queries (hyena / leopard); ~40–45% of test images are closer
  to another test image than to any train image. Expect a large `new_individual` share
  (plausibly 30–50%) — the new decision is likely the biggest lever.
- MiewID cosine k-NN (centred + L2, max-sim per identity), LOO on train, known ids only:
  MAP@5 0.88 hyena / 0.89 leopard.
- Same-id similarity collapses across left/right flanks (~0.15 vs ~0.4 same viewpoint), almost
  to the different-id level → use viewpoint.
- First-pass SIFT ratio-test + homography RANSAC is weak (pair AUC 0.62 / 0.70); needs a better
  geometric model before judging.

## Current best and next ideas

- **Public LB:** best team score is 0.83484 (b1's notebook on branch `submissions/b1`);
  our `combo_lda_sift` got 0.80145.
- **Validation is now split v2 (D-013)**, which includes known animals seen only from the
  other flank. v1 (`cross_frac=0`) was over-optimistic (D-012). Report v2 numbers, honest
  tuning (choose on 2 folds, score the 3rd).
- `models/sideaware.py` scored 0.8553 on v2 but **did worse on the LB than combo (D-014)** — v2's
  cross-view share is probably too high. Our best LB is still `combo_lda_sift.csv` (0.80145);
  team best is b1's notebook (0.83484). Prefer changes that help on both v1 and v2.
- **Team base model: b1's pipeline, `models/b1_pipeline.py` (D-015).** On v2: 0.8308 (non-xview
  0.8956) vs our combo 0.8145 (0.8830). Judge changes on the **non-xview v2 metric** + LB.
  Rank-fusing our models into it adds nothing. **Best variant: wider SIFT (D-016)**, run with
  `B1_SIFT_REFS=4 B1_TAIL_REFS=2 B1_SIFT_TOP2=1 B1_OUT=submissions/b1_refs4 python -m models.b1_pipeline`
  (non-xview 0.8994 vs 0.8956) — awaiting LB check.
- v1-era gains (`sift2_*`, `trans_graph` k-reciprocal) have not been re-checked on v2 yet.
- Next: port sift2 coverage + k-reciprocal onto sideaware and re-tune on v2;
  cross-flank candidate signals (identities lacking the query flank are under-represented in the
  top-10 SIFT candidates); study b1's viewpoint-weighted LambdaRank/novelty models.
- Teammate branches: `submissions/b1` (0.835 notebook), `amigo` (separate history; `src/`
  experiments incl. LOBO KDE new-rate priors).

## Environment notes

- All data is extracted and verified complete (2026-11-22): every train/test id has both
  SIFT `.npy` files; some images have 0 keypoints.
- Python 3.13, numpy 2.4, pandas 3.1, scikit-learn 1.9, opencv 4.10 are installed on the
  original author's machine. xgboost / catboost / faiss are **not** installed there; lightgbm 4.7 is (installed 2026-11-23 for b1's pipeline).
- See `requirements.txt` for the allowed package set.
