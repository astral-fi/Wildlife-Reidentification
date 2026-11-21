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
- **Status:** active

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
- **Status:** proposed

### D-004 — Data lives in `projectwild3/` (full dataset)
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Supersedes:** the path in D-002.
- **Decision:** The full dataset `projectwild3.zip` is unzipped at the repo root as
  `projectwild3/` and git-ignored (1.2 GB). CSVs and MiewID descriptors are in
  `projectwild3/reid_student_data_main/`; SIFT features are in
  `projectwild3/sift_features_{train,test}.npz/` as directories of per-image `.npy` files.
  Code reads data from these paths by default. Exact layout is in `CLAUDE.md`.
- **Why:** this is the complete download (includes the SIFT features that were missing).
- **Status:** active

### D-005 — Never use the training images
- **Date:** 2026-11-22 · **Author:** astral-fi via Claude
- **Decision:** `projectwild3/train_images_*` contain the training cut-out JPEGs. No code may
  read them to make features or train anything.
- **Why:** the rules forbid computing new features from images (academic misconduct).
- **Status:** active

---

## Open questions (for the humans / TA)

- Are we allowed to *look* at the provided training images (e.g. to sanity-check SIFT matches
  or debug), as long as nothing computed from them goes into a model? Until answered: no.
