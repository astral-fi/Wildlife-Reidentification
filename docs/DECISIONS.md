# Decision log

Append-only. Newest at the bottom. Each entry: id, date, author (person + "via Claude" if an
agent made it), decision, why, status. To reverse a decision, add a new entry with
`Supersedes: D-xxx` and set the old one's status to `superseded by D-yyy` (the only edit
allowed to an old entry).

---

### D-001 — Shared context lives in the repo
- **Date:** 2026-11-21 · **Author:** astral-fi via Claude
- **Decision:** `CLAUDE.md` (auto-loaded by Claude Code) holds the problem statement, rules
  and conventions. `docs/DECISIONS.md` (this file) logs decisions; `docs/EXPERIMENTS.md` logs
  scored runs. Every agent reads all three before working and appends to them after.
- **Why:** several teammates use separate Claude agents; this is how they share state when
  they pull the repo.
- **Status:** superseded by D-006 (file name only)

### D-002 — Data stays out of git
- **Date:** 2026-11-21 · **Author:** astral-fi via Claude
- **Decision:** Competition data lives in `reid_student_data_main/` at the repo root (the
  folder name it unzips to) and is git-ignored, as are `*.npy`, `*.npz`, `cache/` and
  `submissions/`. Code should read data from `reid_student_data_main/` by default.
- **Why:** `miewid_features.npy` is 82 MB and the SIFT `.npz` files are bigger; GitHub warns
  above 50 MB and rejects above 100 MB. Everyone has the data from the course page anyway.
- **Status:** path part superseded by D-004 (data stays out of git still holds)

### D-003 — Build our own validation split (proposal, not implemented yet)
- **Date:** 2026-11-21 · **Author:** astral-fi via Claude
- **Decision:** There is no validation file, so validate on a split of `train.csv` that mimics
  test: per species, hold out a set of **whole identities** (their images become
  `new_individual` queries) plus some images of the remaining identities (known-id queries).
  Score with MAP@5 exactly as the competition does. Use a fixed seed and share the split code
  so all teammates report comparable numbers.
- **Why:** the `new_individual` decision is a large part of the score and can only be tuned if
  validation contains unseen identities. Random image-level splits would overstate accuracy.
- **Open:** fraction of unseen identities. Test has a much higher hyena share than train
  (44% vs 27%), which hints at many new hyenas; the public LB is the only check on this.
- **Status:** implemented as D-009

### D-004 — Data lives in `projectwild3/` (full dataset)
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Supersedes:** the path in D-002.
- **Decision:** The full dataset `projectwild3.zip` is unzipped at the repo root as
  `projectwild3/` and git-ignored (1.2 GB). CSVs and MiewID descriptors are in
  `projectwild3/reid_student_data_main/`; SIFT features are in
  `projectwild3/sift_features_{train,test}.npz/` as directories of per-image `.npy` files.
  Code reads data from these paths by default. Exact layout is in `AGENTS.md`.
- **Why:** this is the complete download (includes the SIFT features that were missing).
- **Status:** active

### D-005 — Never use the training images
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Decision:** `projectwild3/train_images_*` contain the training cut-out JPEGs. No code may
  read them to make features or train anything.
- **Why:** the rules forbid computing new features from images (academic misconduct).
- **Status:** active

### D-006 — Context is agent-neutral: `AGENTS.md`
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Supersedes:** D-001 (file name only).
- **Decision:** All shared context moved from `CLAUDE.md` to `AGENTS.md`, the cross-tool
  convention read by Codex, Cursor, Aider, Jules, etc. `CLAUDE.md` and `GEMINI.md` import it
  (`@AGENTS.md`); `.github/copilot-instructions.md` points to it. Every agent must keep
  `AGENTS.md`, `DECISIONS.md` and `EXPERIMENTS.md` up to date as part of its work.
- **Why:** teammates use different agents; one file avoids drifting copies.
- **Status:** active

### D-007 — Default descriptor preprocessing: centre on train mean, then L2-normalise
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Decision:** Compare MiewID descriptors by cosine after subtracting the train mean and
  L2-normalising each row. Work per species.
- **Why:** EDA (`notebooks/01_eda.ipynb` §2) — rows aren't normalised and dimensions have
  non-zero means; centring raised same-vs-different pair AUC (0.857→0.864 hyena,
  0.874→0.877 leopard). Small gain, but it's the baseline everyone should start from.
- **Status:** active

### D-008 — Treat `new_individual` as the main problem; validation must reflect a high new-rate
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Decision:** Prioritise the new-individual decision (thresholds, calibrated P(new), test
  clustering). When D-003's split is implemented, hold out enough identities that roughly
  30–50% of validation queries are new, and report MAP@5 at a couple of new-rates.
- **Why:** EDA §4 — test's nearest-train similarities look much more like never-seen
  identities than train-LOO does. Leave-one-out on train has only ~1.4% new queries, so a
  threshold tuned there would almost never say new (EDA §3 plot).
- **Status:** active

### D-009 — Shared validation split and evaluation harness (`reid/`)
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Implements:** D-003, D-008.
- **Decision:** All models are scored with `reid.evaluate.evaluate(predict)` on
  `reid.split.make_split(seed)` for seeds 0,1,2 (3 folds), reporting mean MAP@5 overall, per
  species, and on known / new queries. Per species: whole identities with ≤20 images are held
  out until ~40% of queries are new (actual ≈44%), plus 25% of the images of each remaining
  identity with ≥2 images as known queries; the rest is the gallery. Models get the gallery
  (with labels) and queries (without) through one `predict` interface, and the same function
  produces test submissions via `reid.evaluate.make_submission`.
- **Why:** comparable numbers across teammates and agents; a realistic new-rate (D-008).
- **Caveat:** the new-rate (~44%) is a guess for test. Robust methods should also be checked
  with `evaluate(..., new_frac=0.25)` and `new_frac=0.55`.
- **Status:** active

### D-010 — Current best approach: cos+LDA blend + SIFT affine re-ranking (`models/combo_lda_sift.py`)
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude (four parallel agents + combination)
- **Decision:** The best pipeline so far, per species:
  1. Identity score = max over that identity's gallery images of the mean of (a) cosine on
     centred+L2 MiewID and (b) cosine in a PCA-256 → LDA-128 (shrinkage 0.3) space fitted on
     the gallery (re-centred, L2). (`models/metric_learning.py`)
  2. For the top-10 cosine identities, SIFT-verify the query against each identity's 2
     best-cosine images: RootSIFT, mutual-NN + 0.8 ratio, `cv2.estimateAffinePartial2D`
     RANSAC 8 px. Add `0.09 * log1p(max(inliers - 3, 0))` (max over the 2 images).
     (`models/sift_match.py`, `models/sift_rerank.py`; pair results cached in `cache/`.)
  3. `new_individual` first if the top score < T (hyena 0.495, leopard 0.53), else second.
- **Why:** the two parts are complementary — LDA blend mostly improves the new decision,
  SIFT mostly improves known-id ranking (+0.027 known). Combined val 0.9235 ±0.004 vs
  baseline 0.9026; each part alone 0.909 / 0.918.
- **Caveat:** T and w were tuned on the same 3 folds they are scored on (slightly optimistic).
- **Status:** active

### D-011 — Findings that closed off directions (don't redo without a new idea)
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Decision / findings:**
  - Calibrated P(new) via gallery-trained logistic regression + expected-AP placement
    (`models/newid_lr.py`): 0.9027, no gain over a tuned threshold. New queries already score
    ~0.97; the remaining loss is mostly known-id ranking.
  - Learned candidate re-ranker (logistic regression on 16 cosine/viewpoint features,
    `models/fusion_rerank.py`): 0.9073; HistGradientBoosting no better than LR. P(new) as
    `1 - max prob` beats `1 - sum prob`. Not yet combined with SIFT/LDA features — a natural
    next step (add SIFT inliers and LDA score as features).
  - LDA or PCA whitening alone, query expansion (k=3,5), top-2 identity aggregation, and
    simple additive viewpoint bonuses/penalties all hurt slightly (0.001–0.003).
  - SIFT: geometric model choice barely matters once matches are mutual-NN filtered
    (affine ≈ homography 20 px ≈ match count); fundamental matrix slightly worse. SIFT is
    weaker than cosine for the new decision (AUC 0.75 vs 0.90 hyena).
- **Status:** active

---

## Open questions (for the humans / TA)

- Are we allowed to *look* at the provided training images (e.g. to sanity-check SIFT matches
  or debug), as long as nothing computed from them goes into a model? Until answered: no.
