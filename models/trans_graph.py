"""Transductive / graph methods on top of combo_lda_sift (agent: trans).

Per species we build the cos+LDA blended similarity (same as metric_learning.embed, D-010)
for gallery+query jointly, plus the combo's SIFT inlier bonus on its top-10 candidates
(read from cache/sift_pairs.pkl, no new pairs). On top of the base identity score
    S0[q, id] = max_g blend(q, g) + w * log1p(max(inl - 3, 0))
three transductive steps (each optional):
  1. k-reciprocal re-ranking (Zhong et al. 2017): blend' = (1-lam)*blend + lam*(1 - Jaccard)
  2. query clustering: components of the query graph with edges blend(q,q') > tau (mutual top-m);
     score = (1-beta)*own + beta*cluster-max; new decision on cluster-pooled top-1.
  3. pseudo-labelling: queries whose top-1 >= Tc join the gallery of their top identity;
     other queries' scores = max(old, gamma * blend(q, pseudo)).
new_individual first if top-1 < T[species], else second (as combo).

Run: OMP_NUM_THREADS=4 python -m models.trans_graph
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from models.sift_match import load_cache, pair_stats
from models.sift_rerank import candidates, inl_value
from reid.data import NEW, l2n, load
from reid.metric import map5
from reid.split import N_FOLDS, make_split

W_SIFT, C_SIFT = 0.09, 3


_CACHE = None


def compute(pairs, workers=4):
    """Read-only use of cache/sift_pairs.pkl; missing pairs are matched in memory only."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_cache()
    todo = sorted({p for p in pairs if p not in _CACHE})
    if todo:
        import cv2
        from multiprocessing import Pool
        cv2.setNumThreads(1)
        print(f"matching {len(todo)} uncached pairs (not saved)", flush=True)
        with Pool(workers) as pool:
            for p, s in pool.map(pair_stats, todo, chunksize=64):
                _CACHE[p] = s
    return {p: _CACHE[p] for p in pairs}


def blend_sims(Xg, Xq, yg):
    mu = Xg.mean(0)
    Fg, Fq = l2n(Xg - mu), l2n(Xq - mu)
    k = min(128, len(np.unique(yg)) - 1)
    p = PCA(256, random_state=0).fit(Fg)
    Pg, Pq = p.transform(Fg), p.transform(Fq)
    lda = LDA(solver="eigen", shrinkage=0.3, n_components=k).fit(Pg, yg)
    Lg, Lq = lda.transform(Pg), lda.transform(Pq)
    m = Lg.mean(0)
    Lg, Lq = l2n(Lg - m), l2n(Lq - m)
    A = np.vstack([Fg, Fq]).astype(np.float32)
    L = np.vstack([Lg, Lq]).astype(np.float32)
    return (A @ A.T + L @ L.T) / 2  # (G+Q)^2 blended similarity


def prepare(gal, Xg, qry, Xq):
    """Per species dict: S (N,N) blend sim gallery-then-query, G, inv (gallery id index),
    uniq, B (Q,U) SIFT bonus, qidx (positions in qry)."""
    cands, pairs = candidates(gal, Xg, qry, Xq)
    stats = compute(pairs)
    out = {}
    for s in ("hyena", "leopard"):
        gm = (gal.species == s).values
        qm = np.where((qry.species == s).values)[0]
        yg = gal.identity.values[gm]
        S = blend_sims(Xg[gm], Xq[qm], yg)
        uniq, inv = np.unique(yg, return_inverse=True)
        col = {u: j for j, u in enumerate(uniq)}
        B = np.zeros((len(qm), len(uniq)), np.float32)
        for r, q in enumerate(qm):
            qid, cand = cands[q]
            for l, _, gs in cand:
                x = max(inl_value(stats[(qid, g)], "aff") for g in gs)
                B[r, col[l]] = W_SIFT * np.log1p(max(x - C_SIFT, 0))
        out[s] = dict(S=S, G=int(gm.sum()), inv=inv, uniq=uniq, B=B, qidx=qm)
    return out


def kreciprocal_jaccard(S, G, k1=20, k2=6):
    """Zhong et al. Jaccard distance of query rows (index >= G) vs all rows."""
    N = len(S)
    D = np.clip(2 - 2 * S, 0, None)
    nn = np.argpartition(D, k1 + 1, axis=1)[:, : k1 + 1]
    nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(D, nn, 1), 1), 1)
    half = nn[:, : int(round(k1 / 2)) + 1]
    sets = [set(r) for r in nn]
    halfsets = [set(r) for r in half]
    V = np.zeros((N, N), np.float32)
    for i in range(N):
        kr = [j for j in nn[i] if i in sets[j]]
        exp = set(kr)
        for j in kr:
            cj = [c for c in half[j] if j in halfsets[c]]
            if len(set(cj) & set(kr)) > 2 / 3 * len(cj):
                exp |= set(cj)
        e = np.fromiter(exp, int)
        w = np.exp(-D[i, e])
        V[i, e] = w / w.sum()
    if k2 > 1:  # local query expansion
        nk2 = nn[:, :k2]
        V2 = np.empty_like(V)
        for i in range(N):
            V2[i] = V[nk2[i]].mean(0)
        V = V2
    J = np.empty((N - G, N), np.float32)
    for r, i in enumerate(range(G, N)):
        idx = np.nonzero(V[i])[0]
        mn = np.minimum(V[i, idx][None, :], V[:, idx]).sum(1)
        J[r] = 1 - mn / (2 - mn)
    return J


def base_scores(d, Sq):
    """Sq: (Q, N) query-vs-all similarity -> (Q,U) identity max score + SIFT bonus."""
    G, inv, U = d["G"], d["inv"], len(d["uniq"])
    M = np.full((len(Sq), U), -np.inf, np.float32)
    np.maximum.at(M.T, inv, Sq[:, :G].T)
    return M + d["B"]


def scores(d, p):
    S, G = d["S"], d["G"]
    Sq = S[G:]
    if p.get("lam", 0) > 0:
        key = ("J", p.get("k1", 20), p.get("k2", 6))
        if key not in d:
            d[key] = kreciprocal_jaccard(S, G, key[1], key[2])
        Sq = (1 - p["lam"]) * Sq + p["lam"] * (1 - d[key])
    M = base_scores(d, Sq)
    Sqq = Sq[:, G:].copy()
    np.fill_diagonal(Sqq, -np.inf)
    # 3. pseudo-labelling (one round)
    if p.get("Tc") is not None:
        top = M.argmax(1)
        t1 = M.max(1)
        conf = np.where(t1 >= p["Tc"])[0]
        if len(conf):
            add = np.full_like(M, -np.inf)
            np.maximum.at(add.T, top[conf], Sqq[:, conf].T)
            M = np.maximum(M, p.get("gamma", 1.0) * add + 0 * M)
    t1 = M.max(1)
    # 2. clustering
    tn = t1
    if p.get("tau") is not None:
        Q = len(M)
        m = p.get("mut", 3)
        nnq = np.argsort(-Sqq, 1)[:, :m]
        g = nx.Graph()
        g.add_nodes_from(range(Q))
        nset = [set(r) for r in nnq]
        for i in range(Q):
            for j in nnq[i]:
                if Sqq[i, j] > p["tau"] and i in nset[j]:
                    g.add_edge(i, j)
        cmax = M.copy()
        tn = t1.copy()
        for comp in nx.connected_components(g):
            c = np.fromiter(comp, int)
            if len(c) < 2:
                continue
            cm = M[c].max(0)
            cmax[c] = cm
            tc = t1[c]
            pool = {"max": tc.max(), "mean": tc.mean()}[p.get("pool", "max")]
            tn[c] = (1 - p.get("nb", 0.5)) * tc + p.get("nb", 0.5) * pool
        beta = p.get("beta", 0.3)
        M = (1 - beta) * M + beta * cmax
    return M, tn


def build(data, nq, p, T):
    out = [None] * nq
    for s, d in data.items():
        M, tn = scores(d, p)
        o = np.argsort(-M, 1)[:, :5]
        for r, q in enumerate(d["qidx"]):
            tp = list(d["uniq"][o[r]])
            out[q] = [NEW] + tp[:4] if tn[r] < T[s] else tp[:1] + [NEW] + tp[1:4]
    return out


TGRID = np.arange(0.40, 0.66, 0.005)


def tune_T(folds, p, fi):
    T = {}
    for s in ("hyena", "leopard"):
        res = []
        for f in fi:
            data, truth = folds[f]
            M, tn = scores(data[s], p)
            o = np.argsort(-M, 1)[:, :5]
            tp = data[s]["uniq"][o]
            tr = truth[data[s]["qidx"]]
            hit = (tp == tr[:, None])
            # AP if new first: 1 if new else 1/(rank+2); if new second: rank0 ->1, new ->1/2, else 1/(rank+2)
            rk = np.where(hit.any(1), hit.argmax(1), 99)
            isnew = tr == NEW
            ap_nf = np.where(isnew, 1.0, np.where(rk < 4, 1 / (rk + 2), 0))
            ap_ns = np.where(isnew, 0.5, np.where(rk == 0, 1.0, np.where(rk < 4, 1 / (rk + 2), 0)))
            res.append((tn, ap_nf, ap_ns, len(truth)))
        def sc(t):
            return np.mean([np.where(tn < t, a, b).sum() / n for tn, a, b, n in res])
        T[s] = max(TGRID, key=sc)
    return T


def evalp(folds, p, fi, T):
    return [map5(build(folds[f][0], len(folds[f][1]), p, T), folds[f][1]) for f in fi]


def load_folds(**kw):
    train, _, Xtr, _ = load()
    folds = []
    for f in range(N_FOLDS):
        g, q, t = make_split(f, **kw)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        folds.append((prepare(gal, Xtr[g], qry, Xtr[q]), t))
    return folds


def honest(folds, grid, label):
    """Pick (p, T) on 2 folds, score the 3rd. Also report in-sample best."""
    tuned = {}
    for k, p in enumerate(grid):
        for hold in range(N_FOLDS):
            tr = [f for f in range(N_FOLDS) if f != hold]
            T = tune_T(folds, p, tr)
            tuned[(k, hold)] = (np.mean(evalp(folds, p, tr, T)), T)
        Tall = tune_T(folds, p, range(N_FOLDS))
        ins = evalp(folds, p, range(N_FOLDS), Tall)
        print(f"  {label} {p} in-sample {np.mean(ins):.4f}", flush=True)
    hs = []
    for hold in range(N_FOLDS):
        k = max(range(len(grid)), key=lambda k: tuned[(k, hold)][0])
        hs.append(evalp(folds, grid[k], [hold], tuned[(k, hold)][1])[0])
        print(f"  {label} hold {hold}: pick {grid[k]} -> {hs[-1]:.4f}", flush=True)
    print(f"{label} HONEST {np.mean(hs):.4f} ±{np.std(hs, ddof=1):.4f}", flush=True)
    return np.mean(hs)


if __name__ == "__main__" and "--submit" not in sys.argv:
    folds = load_folds()
    which = sys.argv[1:] or ["base", "kr", "cl", "pl"]
    if "base" in which:
        honest(folds, [{}], "combo")
    if "kr" in which:
        honest(folds, [dict(lam=l, k1=k1, k2=k2) for l in (0.1, 0.2, 0.3) for k1, k2 in ((10, 3), (20, 6))], "kr")
    if "cl" in which:
        honest(folds, [dict(tau=t, beta=b, nb=nb, mut=3) for t in (0.5, 0.6, 0.7)
                       for b in (0, 0.2, 0.4) for nb in (0, 0.5)], "cl")
    if "pl" in which:
        honest(folds, [dict(Tc=tc, gamma=g) for tc in (0.6, 0.7, 0.8) for g in (0.9, 1.0)], "pl")


def robustness(p, new_fracs=(0.25, 0.55)):
    """Combo vs p at other new-rates, T tuned in-sample per setting (same for both)."""
    for nf in new_fracs:
        fo = load_folds(new_frac=nf)
        for lab, q in (("combo", {}), ("trans", p)):
            T = tune_T(fo, q, range(N_FOLDS))
            print(f"new_frac={nf} {lab} in-sample {np.mean(evalp(fo, q, range(N_FOLDS), T)):.4f}"
                  f" T={ {k: round(float(v), 3) for k, v in T.items()} }", flush=True)
        honest(fo, [{}], f"combo@{nf}")
        honest(fo, [p], f"trans@{nf}")


BEST = dict(lam=0.2, k1=10, k2=3)
BEST_T = None  # filled by tune on all 3 folds in submit()


def make_predict(p, T):
    def predict(gal, Xg, qry, Xq):
        return build(prepare(gal, Xg, qry, Xq), len(qry), p, T)
    return predict


def submit():
    from reid.evaluate import make_submission
    folds = load_folds()
    T = tune_T(folds, BEST, range(N_FOLDS))
    print("T", T, "in-sample", np.mean(evalp(folds, BEST, range(N_FOLDS), T)), flush=True)
    del folds
    make_submission(make_predict(BEST, T), "trans_best")


if __name__ == "__main__" and "--submit" in sys.argv:
    submit()
