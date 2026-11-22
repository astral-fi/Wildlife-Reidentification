"""Shared validation split (D-009, implements D-003/D-008).

Per species, from train.csv:
- **unseen queries**: every image of a random set of held-out identities (their truth becomes
  `new_individual`). Only identities with <= MAX_UNSEEN_ID_SIZE images are eligible, so one huge
  identity can't dominate.
- **known queries**: KNOWN_FRAC of the images of each remaining identity that has >= 2 images
  (at least one image always stays in the gallery).
- Unseen identities are added until unseen queries are ~NEW_FRAC of all queries.
- **cross-view queries (v2, D-013)**: for each remaining identity whose gallery images show
  both flanks (left-ish and right-ish), with probability CROSS_FRAC move every gallery image of
  one random flank into the queries. These are known animals seen only from the other side in
  the gallery — common in test (D-012). Pass cross_frac=0 to get the old v1 split.
- Gallery = everything else.

Folds are seeds 0..N_FOLDS-1. Same seed → same split on every machine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NEW, load

NEW_FRAC = 0.40
KNOWN_FRAC = 0.25
MAX_UNSEEN_ID_SIZE = 20
N_FOLDS = 3
CROSS_FRAC = 0.5
SIDE = {'left': 'L', 'frontleft': 'L', 'backleft': 'L',
        'right': 'R', 'frontright': 'R', 'backright': 'R'}


def make_split(seed: int = 0, new_frac: float = NEW_FRAC, known_frac: float = KNOWN_FRAC,
               cross_frac: float = CROSS_FRAC):
    """Return (gallery_idx, query_idx, query_truth) as index arrays into train / truth labels."""
    train = load()[0]
    rng = np.random.default_rng(seed)
    gal, qry, truth = [], [], []
    for s in ("hyena", "leopard"):
        sub = train[train.species == s]
        groups = {k: v.index.values for k, v in sub.groupby("identity")}
        names = np.array(sorted(groups))
        rng.shuffle(names)
        # known queries first (from ids not held out); decide unseen ids greedily
        n_known_est = int(known_frac * sum(len(v) for v in groups.values() if len(v) >= 2))
        target_unseen = int(round(n_known_est * new_frac / (1 - new_frac)))
        unseen, n_unseen = set(), 0
        for n in names:
            if n_unseen >= target_unseen:
                break
            if len(groups[n]) <= MAX_UNSEEN_ID_SIZE:
                unseen.add(n)
                n_unseen += len(groups[n])
        for n in names:
            idx = groups[n]
            if n in unseen:
                qry += list(idx); truth += [NEW] * len(idx)
            elif len(idx) >= 2:
                k = min(len(idx) - 1, max(1, int(round(known_frac * len(idx)))))
                pick = rng.choice(idx, k, replace=False)
                qry += list(pick); truth += [n] * k
                gal += [i for i in idx if i not in set(pick)]
            else:
                gal += list(idx)
    gal, qry, truth = np.array(gal), np.array(qry), np.array(truth, dtype=object)
    if cross_frac > 0:
        crng = np.random.default_rng(10_000 + seed)  # separate stream: v1 part is unchanged
        side = train.viewpoint.map(SIDE).fillna("O").values
        gdf = train.loc[gal]
        moved = []
        for n, grp in sorted(gdf.groupby("identity"), key=lambda x: x[0]):
            sd = side[grp.index]
            if (sd == "L").any() and (sd == "R").any() and crng.random() < cross_frac:
                moved += list(grp.index[sd == crng.choice(["L", "R"])])
        moved = np.array(moved, dtype=int)
        gal = np.setdiff1d(gal, moved)
        qry = np.r_[qry, moved]
        truth = np.r_[truth, train.identity.values[moved]].astype(object)
    return gal, qry, truth


def cross_view_mask(seed: int = 0, **kw):
    """Boolean mask over make_split(seed)'s queries: True for v2 cross-view queries."""
    g, q, t = make_split(seed, cross_frac=0.0, **{k: v for k, v in kw.items() if k != "cross_frac"})
    _, q2, _ = make_split(seed, **kw)
    return np.r_[np.zeros(len(q), bool), np.ones(len(q2) - len(q), bool)]


def describe(seed: int = 0) -> pd.DataFrame:
    train = load()[0]
    g, q, t = make_split(seed)
    out = []
    for s in ("hyena", "leopard"):
        m = train.species.values[q] == s
        out.append(dict(species=s, gallery=int((train.species.values[g] == s).sum()),
                        queries=int(m.sum()), new_frac=float(np.mean(t[m] == NEW)),
                        cross_view=int((cross_view_mask(seed)[m]).sum())))
    return pd.DataFrame(out)
