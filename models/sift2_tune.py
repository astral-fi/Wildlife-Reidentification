"""Tune SIFT usage inside the cos+LDA combo on cached sift2 features; honest 2-fold-tune / 1-fold-score.

Per identity: score = blend + w_s * log1p(max(agg - c, 0)), agg over the first P verified images
(best cosine first, optionally same/unknown-viewpoint images first), agg = max or sum of top-2
inliers (raw affine count or count/sqrt(min kp)*10). Candidates: top-K ids by cosine or blend.
new_individual first iff top score < T_s and top inliers < H_s, else second.

Run: python -m models.sift2_tune   (needs cache/sift2_feats.pkl from models.sift2_feats)
"""
import itertools
import pickle

import numpy as np

from reid.data import NEW
from models.sift2_feats import FEATS

FLIP = {"left": "right", "right": "left", "frontleft": "frontright", "frontright": "frontleft",
        "backleft": "backright", "backright": "backleft"}
TS = np.arange(0.40, 0.62, 0.005)
HS = [np.inf, 15, 25, 40, 60]


def ap(top, t):
    return 1.0 / (top.index(t) + 1) if t in top[:5] else 0.0


def per_query(F, truth, struct, w, c):
    """Arrays: top score, top agg inliers, AP if new-first, AP if new-second."""
    mode, k, p, vp, agg, norm = struct
    out = np.zeros((len(F), 4))
    for i, ((s, qvp, cand), t) in enumerate(zip(F, truth)):
        key = (lambda x: -x[1]) if mode == "blend" else (lambda x: -x[2])
        sc = []
        for l, b, cmax, ims in sorted(cand, key=key)[:k]:
            if vp and isinstance(qvp, str):
                ims = sorted(ims, key=lambda m: (isinstance(m[4], str) and m[4] != qvp, -m[0]))
            v = [(a / np.sqrt(max(min(na, nb), 1)) * 10 if norm else a) for _, a, na, nb, _ in ims[:p]]
            v.sort(reverse=True)
            x = v[0] + (v[1] if agg == "sum2" and len(v) > 1 else 0)
            sc.append((b + w * np.log1p(max(x - c, 0)), x, l))
        sc.sort(reverse=True)
        top = [l for _, _, l in sc]
        out[i] = (sc[0][0], sc[0][1], ap([NEW] + top[:4], t), ap(top[:1] + [NEW] + top[1:4], t))
    return out


STRUCTS = [(m, k, p, vp, a, n) for m in ("cos", "blend") for k in (10, 15) for p in (1, 2, 3, 5)
           for vp in (0, 1) for a in ("max", "sum2") for n in (0, 1)]
WS = {0: [0.03, 0.06, 0.09, 0.12, 0.15], 1: [0.03, 0.06, 0.09, 0.12, 0.15]}
CS = {0: [0, 3, 6], 1: [0, 2, 4]}


SUB = None


def _job(j):
    sp, st, w, c = j
    return [per_query(F, t, st, w, c) for F, t in SUB[sp]]


def best_TH(arrs, hs=None):
    """arrs: list of per-query arrays (one per fold); return (score, T, H) maximising mean MAP."""
    best = (-1, None, None)
    for H in (hs or HS):
        for T in TS:
            m = np.mean([np.where((a[:, 0] < T) & (a[:, 1] < H), a[:, 2], a[:, 3]).mean() for a in arrs])
            if m > best[0]:
                best = (m, T, H)
    return best


def apply(a, T, H):
    return np.where((a[:, 0] < T) & (a[:, 1] < H), a[:, 2], a[:, 3])


if __name__ == "__main__":
    folds = pickle.load(open(FEATS, "rb"))
    from multiprocessing import Pool
    SUB = {sp: [([x for x in F if x[0] == sp], t[[x[0] == sp for x in F]]) for F, t in folds]
           for sp in ("hyena", "leopard")}
    jobs = [(sp, st, w, c) for sp in SUB for st in STRUCTS
            for w, c in itertools.product(WS[st[5]], CS[st[5]])]
    with Pool(6) as pool:
        res = dict(zip(jobs, pool.map(_job, jobs, chunksize=8)))
    pickle.dump(res, open(FEATS.with_name("sift2_grid.pkl"), "wb"))
