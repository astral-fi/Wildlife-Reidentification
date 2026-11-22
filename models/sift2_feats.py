"""SIFT-in-combo, wider coverage: per query keep the union of the top-K cosine and top-K blend
(cos+LDA) identities, and verify up to P best-cosine gallery images per identity.
Pair stats come from models.sift_match.pair_stats (provided keypoints only); own cache
cache/sift2_pairs.pkl. Output per query: species, viewpoint, list of
(identity, blend, maxcos, [(cos, aff, n_a, n_b, vp_gallery), ...] sorted by cos desc).

Run: python -m models.sift2_feats   (builds val features for 3 folds -> cache/sift2_feats.pkl)
"""
import pickle
import time
from multiprocessing import Pool

import cv2
import numpy as np

from models.metric_learning import embed, id_scores
from models.sift_match import pair_stats
from reid.data import ROOT, load, preprocess
from reid.split import N_FOLDS, make_split

CACHE = ROOT / "cache" / "sift2_pairs.pkl"
FEATS = ROOT / "cache" / "sift2_feats.pkl"
LDA_CFG = dict(lda=128, shrink=0.3)
K, P = 15, 5


def compute(pairs, workers=6):
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    todo = sorted({p for p in pairs if p not in cache})
    if todo:
        cv2.setNumThreads(1)
        t0 = time.time()
        print(f"matching {len(todo)} new pairs", flush=True)
        with Pool(workers) as pool:
            for p, s in pool.imap_unordered(pair_stats, todo, chunksize=64):
                cache[p] = s
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"  done {time.time() - t0:.0f}s", flush=True)
    return {p: cache[p] for p in pairs}


def features(gal, Xg, qry, Xq, k=K, p=P):
    Fg, Fq = preprocess(Xg, Xq)
    out = [None] * len(qry)
    pairs = []
    for s in ("hyena", "leopard"):
        gm = np.where((gal.species == s).values)[0]
        qm = np.where((qry.species == s).values)[0]
        Sqg, Sgg = embed(Xg[gm], Xq[qm], gal.identity.values[gm], LDA_CFG)
        uniq, M = id_scores(Sqg, Sgg, gal.iloc[gm], qry.iloc[qm], LDA_CFG)
        S = Fq[qm] @ Fg[gm].T
        labs = gal.identity.values[gm]
        inv = np.searchsorted(uniq, labs)
        C = np.full_like(M, -np.inf)
        np.maximum.at(C.T, inv, S.T)
        gids, gvp = gal.id.values[gm], gal.viewpoint.values[gm]
        for r, q in enumerate(qm):
            ids = set(np.argsort(-M[r])[:k]) | set(np.argsort(-C[r])[:k])
            cand = []
            for j in ids:
                idx = np.where(inv == j)[0]
                idx = idx[np.argsort(-S[r, idx])][:p]
                cand.append((uniq[j], float(M[r, j]), float(C[r, j]),
                             [(float(S[r, i]), gids[i], gvp[i]) for i in idx]))
            qid = qry.id.values[q]
            out[q] = (qid, s, qry.viewpoint.values[q], cand)
            pairs += [(qid, g) for c in cand for _, g, _ in c[3]]
    st = compute(pairs)
    res = []
    for qid, s, vp, cand in out:
        res.append((s, vp, [(l, b, c, [(cs, st[(qid, g)][4], st[(qid, g)][0], st[(qid, g)][1], v)
                                       for cs, g, v in ims]) for l, b, c, ims in cand]))
    return res, len(pairs)


if __name__ == "__main__":
    train, _, Xtr, _ = load()
    folds = []
    for f in range(N_FOLDS):
        t0 = time.time()
        g, q, t = make_split(f)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        F, n = features(gal, Xtr[g], qry, Xtr[q])
        print(f"fold {f}: {len(q)} queries, {n} pairs, {time.time() - t0:.0f}s", flush=True)
        folds.append((F, t))
    pickle.dump(folds, open(FEATS, "wb"))
