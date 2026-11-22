"""fusion2: learned LR re-ranker over cos + LDA-blend + SIFT features (per candidate identity).

Candidates = combo_lda_sift's (top-10 cosine ids x 2 best-cosine images). Per (query, candidate)
features: cosine max/mean/rank/gap, LDA-blend score/gap, viewpoint-matched cosine, SIFT max and
second affine inliers over the 2 verified images, inliers / sqrt(min kp), ratio matches, query
kp count, SIFT-available flag, plus query-level context (top1/top2 blend, max inliers overall).
Training data: pseudo-queries simulated ONLY from the gallery passed to predict (held-out ids ->
new, held-out images -> known), with their own LDA fitted on the reduced gallery and their own
SIFT pairs. Classifiers: LR for P(candidate is true id); LR for P(new) on query-level features.
new_individual placement: (a) alpha*(1-max p), or (b) separate P(new) model: beta*pnew.
Pair cache: cache/fusion2_pairs.pkl (seeded read-only from cache/sift_pairs.pkl).

Run:  OMP_NUM_THREADS=4 python -m models.fusion2_rerank [--submit]
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.metric_learning import FLIP, embed, id_scores
from models.sift_match import CACHE as SHARED_CACHE, pair_stats
from reid.data import NEW, ROOT, load, preprocess
from reid.evaluate import make_submission
from reid.metric import map5
from reid.split import N_FOLDS, make_split

LDA_CFG = dict(lda=128, shrink=0.3)
TOP_ID, PER_ID = 10, 2
PCACHE = ROOT / "cache" / "fusion2_pairs.pkl"
_pc = None


def sift(pairs):
    global _pc
    if _pc is None:
        _pc = {}
        for p in (SHARED_CACHE, PCACHE):
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        _pc.update(pickle.load(f))
                except Exception:
                    pass
    todo = sorted({p for p in pairs if p not in _pc})
    if todo:
        cv2.setNumThreads(1)
        t0 = time.time()
        with Pool(6) as pool:
            for p, s in pool.imap_unordered(pair_stats, todo, chunksize=64):
                _pc[p] = s
        print(f"  matched {len(todo)} pairs in {time.time() - t0:.0f}s", flush=True)
        tmp = str(PCACHE) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(_pc, f)
        os.replace(tmp, PCACHE)
    return _pc


def species_feats(gid, glab, gview, Xg, qid, qview, Xq):
    """One species. Returns feats [nq, k, d], cand labels [nq, k] (object, padded with '')."""
    Fg, Fq = preprocess(Xg, Xq)
    S = Fq @ Fg.T
    import pandas as pd
    gdf = pd.DataFrame({"identity": glab, "viewpoint": gview})
    qdf = pd.DataFrame({"viewpoint": qview})
    Sqg, Sgg = embed(Xg, Xq, glab, LDA_CFG)
    uniq, B = id_scores(Sqg, Sgg, gdf, qdf, LDA_CFG)
    col = {u: j for j, u in enumerate(uniq)}
    cnt = dict(zip(*np.unique(glab, return_counts=True)))
    order = np.argsort(-S, 1)
    cands, pairs = [], []
    for r in range(len(Fq)):
        seen = {}
        for j in order[r]:
            l = glab[j]
            if l not in seen:
                if len(seen) >= TOP_ID:
                    if all(len(v) >= PER_ID for v in seen.values()):
                        break
                    continue
                seen[l] = []
            if len(seen[l]) < PER_ID:
                seen[l].append(j)
        cands.append(seen)
        pairs += [(qid[r], gid[j]) for js in seen.values() for j in js]
    st = sift(pairs)
    k = TOP_ID
    F = np.zeros((len(Fq), k, 20), np.float32)
    C = np.full((len(Fq), k), "", dtype=object)
    members = {}
    for i, l in enumerate(glab):
        members.setdefault(l, []).append(i)
    members = {l: np.array(v) for l, v in members.items()}
    for r, seen in enumerate(cands):
        qv = qview[r]; fv = FLIP.get(qv)
        labs = list(seen)
        cosm = np.array([S[r, members[l]].max() for l in labs])
        bl = np.array([B[r, col[l]] for l in labs])
        sts = [[st[(qid[r], gid[j])] for j in seen[l]] for l in labs]
        aff = [sorted([s[4] for s in ss], reverse=True) + [0] for ss in sts]
        mx_all = max(a[0] for a in aff)
        nq = sts[0][0][0]
        bsort = np.sort(bl)[::-1]
        for c, l in enumerate(labs):
            m = members[l]; s = S[r, m]; vs = gview[m]
            same = vs == qv; opp = (vs == fv) if fv else np.zeros(len(m), bool)
            ss = sts[c]
            nn = [max(min(x[0], x[1]), 1) for x in ss]
            F[r, c] = (cosm[c], s.mean(), c, cosm[0] - cosm[c], bl[c], bsort[0] - bl[c],
                       s[same].max() if same.any() else -1, s[opp].max() if opp.any() else -1,
                       np.log1p(cnt[l]), np.log1p(aff[c][0]), np.log1p(aff[c][1]),
                       max(x[4] / np.sqrt(n) for x, n in zip(ss, nn)),
                       np.log1p(max(x[2] for x in ss)), np.log1p(nq), float(nq >= 50),
                       bsort[0], bsort[0] - bsort[1] if len(bsort) > 1 else 0,
                       np.log1p(mx_all), np.log1p(aff[c][0]) - np.log1p(mx_all),
                       float(qv == "missing"))
            C[r, c] = l
    return F, C


QFEAT = [0, 4, 9, 13, 14, 15, 16, 17]  # query-level features taken from candidate rank 0


def simulate(gid, glab, gview, Xg, rng, new_frac=0.4, known_frac=0.25):
    ids, inv = np.unique(glab, return_inverse=True)
    sizes = np.bincount(inv)
    n_known = int(known_frac * sizes[sizes >= 2].sum())
    target = n_known * new_frac / (1 - new_frac)
    unseen, tot = set(), 0
    for j in rng.permutation(len(ids)):
        if tot >= target:
            break
        if sizes[j] <= 20:
            unseen.add(j); tot += sizes[j]
    q, g = [], []
    for j in range(len(ids)):
        m = np.where(inv == j)[0]
        if j in unseen:
            q += list(m)
        elif len(m) >= 2:
            kk = min(len(m) - 1, max(1, int(round(known_frac * len(m)))))
            p = set(rng.choice(m, kk, replace=False)); q += list(p); g += [x for x in m if x not in p]
        else:
            g += list(m)
    q, g = np.array(q), np.array(g)
    F, C = species_feats(gid[g], glab[g], gview[g], Xg[g], gid[q], gview[q], Xg[q])
    return F, C, glab[q]


def lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


FEAT_CACHE = {}


def fit_predict(gal, Xg, qry, Xq, reps=2, seed=0):
    """Per query: (p_cand [k], labels [k], p_new)."""
    key = (len(gal), len(qry), float(Xq[:3].sum()), reps, seed)
    if key in FEAT_CACHE:
        return FEAT_CACHE[key]
    res = [None] * len(qry)
    for s in ("hyena", "leopard"):
        gm = (gal.species == s).values; qm = np.where((qry.species == s).values)[0]
        gid = gal.id.values[gm]; glab = np.asarray(gal.identity.values[gm], dtype=object)
        gview = np.asarray(gal.viewpoint.values[gm], dtype=object)
        rng = np.random.default_rng(seed)
        Xs, ys, Qs, yn = [], [], [], []
        for _ in range(reps):
            F, C, t = simulate(gid, glab, gview, Xg[gm], rng)
            valid = C != ""
            Xs.append(F[valid]); ys.append((C == t[:, None])[valid])
            Qs.append(F[:, 0, QFEAT]); yn.append(~(C == t[:, None]).any(1))
        clf = lr().fit(np.concatenate(Xs), np.concatenate(ys))
        Fq, Cq = species_feats(gid, glab, gview, Xg[gm], qry.id.values[qm],
                               np.asarray(qry.viewpoint.values[qm], dtype=object), Xq[qm])
        P = clf.predict_proba(Fq.reshape(-1, Fq.shape[-1]))[:, 1].reshape(Cq.shape)
        P[Cq == ""] = -1
        # P(new) model: query-level features + the re-ranker's top probabilities
        Qtr = np.concatenate(Qs)
        nclf = lr().fit(Qtr, np.concatenate(yn))
        pn = nclf.predict_proba(Fq[:, 0, QFEAT])[:, 1]
        for r, q in enumerate(qm):
            res[q] = (P[r], Cq[r], pn[r])
    FEAT_CACHE[key] = res
    return res


def rank(res, mode, a):
    out = []
    for p, c, pn in res:
        ns = a * (1 - p.max()) if mode == "max" else a * pn
        sc = np.append(p, ns); labs = np.append(c, NEW)
        o = [l for l in labs[np.argsort(-sc, kind="stable")] if l != ""]
        out.append(o[:5])
    return out


def make_predict(mode, a):
    return lambda gal, Xg, qry, Xq: rank(fit_predict(gal, Xg, qry, Xq), mode, a)


GRID = {"max": np.round(np.arange(0.3, 3.01, 0.05), 3), "pnew": np.round(np.arange(0.3, 3.01, 0.05), 3)}


def scores(preds, truth, sp):
    r = {"all": map5(preds, truth)}
    for s in ("hyena", "leopard"):
        m = sp == s; r[s] = map5([p for p, k in zip(preds, m) if k], truth[m])
    for n, m in (("known", truth != NEW), ("new", truth == NEW)):
        r[n] = map5([p for p, k in zip(preds, m) if k], truth[m])
    return r


if __name__ == "__main__":
    t00 = time.time()
    train, _, Xtr, _ = load()
    folds = []
    for f in range(N_FOLDS):
        t0 = time.time()
        g, q, t = make_split(f)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        folds.append((fit_predict(gal, Xtr[g], qry, Xtr[q]), t, qry.species.values))
        print(f"fold {f} features+fit {time.time() - t0:.0f}s", flush=True)
    with open(ROOT / "cache" / "fusion2_folds.pkl", "wb") as fh:
        pickle.dump(folds, fh)
    summary = {}
    for mode in ("max", "pnew"):
        # in-sample: best a (per species) on all 3 folds; honest: tune on 2, score 3rd
        def best_a(fs, s):
            return max(GRID[mode], key=lambda a: np.mean(
                [map5(rank([x for x, sp in zip(R, S) if sp == s], mode, a), t[S == s]) for R, t, S in fs]))
        def run(train_fs, test_f):
            A = {s: best_a(train_fs, s) for s in ("hyena", "leopard")}
            R, t, S = test_f
            preds = [None] * len(R)
            for s in A:
                idx = np.where(S == s)[0]
                for i, p in zip(idx, rank([R[i] for i in idx], mode, A[s])):
                    preds[i] = p
            return scores(preds, t, S), A
        ins = [run(folds, folds[k]) for k in range(N_FOLDS)]
        hon = [run([folds[j] for j in range(N_FOLDS) if j != k], folds[k]) for k in range(N_FOLDS)]
        for name, rr in (("in-sample", ins), ("honest", hon)):
            m = {k: np.mean([r[0][k] for r in rr]) for k in rr[0][0]}
            sd = np.std([r[0]["all"] for r in rr], ddof=1)
            print(f"{mode} {name}: " + " ".join(f"{k} {v:.4f}" for k, v in m.items()),
                  f"±{sd:.4f}", "alphas", [r[1] for r in rr], flush=True)
            summary[(mode, name)] = (m, sd, [r[1] for r in rr])
    print(f"total {time.time() - t00:.0f}s")
    if "--submit" in sys.argv:
        mode = sys.argv[sys.argv.index("--submit") + 1]
        A = summary[(mode, "in-sample")][2][0]
        def pred(gal, Xg, qry, Xq):
            res = fit_predict(gal, Xg, qry, Xq)
            out = [None] * len(res)
            for s in A:
                idx = np.where(qry.species.values == s)[0]
                for i, p in zip(idx, rank([res[i] for i in idx], mode, A[s])):
                    out[i] = p
            return out
        make_submission(pred, "fusion2_best")
