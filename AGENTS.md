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

## Environment notes

- All data is extracted and verified complete (2026-11-22): every train/test id has both
  SIFT `.npy` files; some images have 0 keypoints.
- Python 3.13, numpy 2.4, pandas 3.1, scikit-learn 1.9, opencv 4.10 are installed on the
  original author's machine. xgboost / lightgbm / catboost / faiss are **not** installed there.
- See `requirements.txt` for the allowed package set.
