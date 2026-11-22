"""Combination: cos+LDA blended identity score (metric_learning) + SIFT inlier bonus (sift_rerank).

score(id) = blend(id) + w * log1p(max(inliers - c, 0)),  blend = mean of cosine and LDA cosine,
max over the identity's gallery images. Candidates and verified pairs are exactly those of
sift_rerank (top-10 cosine identities, 2 images each), so the SIFT cache is reused.
new_individual first if top score < T[species], else second.

Run:  python -m models.combo_lda_sift [--submit]
"""
import itertools
import sys

import numpy as np

from models.metric_learning import embed, id_scores
from models.sift_match import compute
from models.sift_rerank import candidates, inl_value
from reid.data import NEW, load
from reid.evaluate import evaluate, make_submission
from reid.metric import map5
from reid.split import N_FOLDS, make_split

LDA_CFG = dict(lda=128, shrink=0.3)


def features(gal, Xg, qry, Xq):
    """Per query: (species, [(identity, blend_score, max_inliers)])."""
    cands, pairs = candidates(gal, Xg, qry, Xq)
    stats = compute(pairs)
    blend = {}
    for s in ("hyena", "leopard"):
        gm = (gal.species == s).values
        qm = np.where((qry.species == s).values)[0]
        Sqg, Sgg = embed(Xg[gm], Xq[qm], gal.identity.values[gm], LDA_CFG)
        uniq, M = id_scores(Sqg, Sgg, gal[gm], qry.iloc[qm], LDA_CFG)
        col = {u: j for j, u in enumerate(uniq)}
        for r, q in enumerate(qm):
            blend[q] = (M[r], col)
    out = []
    for q, (qid, cand) in enumerate(cands):
        M, col = blend[q]
        out.append((qry.species.values[q],
                    [(l, float(M[col[l]]), max(inl_value(stats[(qid, g)], "aff") for g in gs))
                     for l, _, gs in cand]))
    return out


def rank(feats, w, c, T):
    preds = []
    for s, cand in feats:
        sc = sorted(((b + w * np.log1p(max(x - c, 0)), l) for l, b, x in cand), reverse=True)
        top = [l for _, l in sc]
        preds.append([NEW] + top[:4] if sc[0][0] < T[s] else top[:1] + [NEW] + top[1:4])
    return preds


def make_predict(w, c, T):
    return lambda gal, Xg, qry, Xq: rank(features(gal, Xg, qry, Xq), w, c, T)


if __name__ == "__main__":
    train, _, Xtr, _ = load()
    folds = []
    for f in range(N_FOLDS):
        g, q, t = make_split(f)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        folds.append((features(gal, Xtr[g], qry, Xtr[q]), t, qry.species.values))
    best = None
    for w, c in itertools.product([0, 0.03, 0.06, 0.09], [3]):
        Ts = {}
        for s in ("hyena", "leopard"):  # tune T per species
            def sc(T):
                return np.mean([map5(rank([fe for fe, sp in zip(F, S) if sp == s], w, c, {s: T}),
                                     t[S == s]) for F, t, S in folds])
            Ts[s] = max(np.arange(0.40, 0.60, 0.005), key=sc)
        allv = [map5(rank(F, w, c, Ts), t) for F, t, _ in folds]
        print(f"w={w} c={c} T={ {k: round(float(v), 3) for k, v in Ts.items()} } all {np.mean(allv):.4f} ±{np.std(allv, ddof=1):.4f}", flush=True)
        if best is None or np.mean(allv) > best[0]:
            best = (np.mean(allv), w, c, Ts)
    print("best", best)
    _, w, c, Ts = best
    evaluate(make_predict(w, c, Ts))
    if "--submit" in sys.argv:
        make_submission(make_predict(w, c, Ts), "combo_lda_sift")
