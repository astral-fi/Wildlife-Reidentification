"""Re-rank MiewID top identities with SIFT verification; SIFT also informs new_individual.

score(id) = maxcos(id) + w * f(max inliers over the id's verified gallery images)
f(x) = log1p(max(x - c, 0))   (c = min-inlier cutoff, noise floor)
new_individual first if  top1 score < T, else at position new_pos.

Run:  python -m models.sift_rerank [--analyse] [--submit]
"""
from __future__ import annotations

import itertools
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from models.sift_match import FIELDS, compute
from reid.data import NEW, load, preprocess
from reid.evaluate import evaluate, make_submission
from reid.metric import map5
from reid.split import N_FOLDS, make_split

TOP_ID = 10   # identities verified per query
PER_ID = 2    # best-cosine gallery images verified per identity


def candidates(gal, Xg, qry, Xq):
    """Per query: list of (identity, maxcos, [gallery ids]) for the top TOP_ID ids + all pairs."""
    Fg, Fq = preprocess(Xg, Xq)
    out = [None] * len(qry)
    pairs = []
    for s in ("hyena", "leopard"):
        gm = np.where((gal.species == s).values)[0]
        qm = np.where((qry.species == s).values)[0]
        labs = gal.identity.values[gm]
        gids = gal.id.values[gm]
        S = Fq[qm] @ Fg[gm].T
        order = np.argsort(-S, axis=1)
        for r, q in enumerate(qm):
            seen = {}
            for j in order[r]:
                l = labs[j]
                if l not in seen:
                    if len(seen) >= TOP_ID and all(len(v[1]) >= PER_ID for v in seen.values()):
                        break
                    if len(seen) >= TOP_ID:
                        continue
                    seen[l] = [float(S[r, j]), []]
                if len(seen[l][1]) < PER_ID:
                    seen[l][1].append(gids[j])
            qid = qry.id.values[q]
            cand = [(l, v[0], v[1]) for l, v in seen.items()]
            out[q] = (qid, cand)
            pairs += [(qid, g) for _, _, gs in cand for g in gs]
    return out, pairs


def inl_value(st, kind):
    d = dict(zip(FIELDS, st))
    if kind == "ratio_norm":
        return d["ratio"] / np.sqrt(max(min(d["n_a"], d["n_b"]), 1))
    if kind == "aff_norm":
        return d["aff"] / np.sqrt(max(min(d["n_a"], d["n_b"]), 1)) * 10
    return d[kind]


def rank(cands, stats, kind, w, c, T, new_pos=2, gate=None):
    preds = []
    for qid, cand in cands:
        sc = []
        for l, cs, gs in cand:
            x = max(inl_value(stats[(qid, g)], kind) for g in gs)
            sc.append((cs + w * np.log1p(max(x - c, 0)), l))
        sc.sort(reverse=True)
        top = [l for _, l in sc]
        if sc[0][0] < T:
            preds.append([NEW] + top[:4])
        else:
            preds.append(top[: new_pos - 1] + [NEW] + top[new_pos - 1: 4])
    return preds


def make_predict(kind="aff", w=0.05, c=4, T=0.45, new_pos=2):
    def predict(gal, Xg, qry, Xq):
        cands, pairs = candidates(gal, Xg, qry, Xq)
        stats = compute(pairs)
        return rank(cands, stats, kind, w, c, T, new_pos)
    return predict


def analyse():
    train, _, Xtr, _ = load()
    folds = []
    t0 = time.time()
    for f in range(N_FOLDS):
        g, q, truth = make_split(f)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        cands, pairs = candidates(gal, Xtr[g], qry, Xtr[q])
        stats = compute(pairs)
        folds.append((cands, stats, truth, qry.species.values))
    print(f"candidates+matching {time.time() - t0:.0f}s", flush=True)

    # --- hard-pair AUC: query vs its top candidate images (same-id vs not) ---
    print("\nHard-pair AUC (query vs gallery images of its top-10 MiewID ids, known queries)")
    kinds = ["ratio", "ratio7", "aff", "fund", "homo", "ratio_norm", "aff_norm"]
    for sp in ("hyena", "leopard"):
        rows = {k: [] for k in kinds + ["cos"]}
        y = []
        for cands, stats, truth, spv in folds:
            for (qid, cand), t, s in zip(cands, truth, spv):
                if s != sp or t == NEW:
                    continue
                for l, cs, gs in cand:
                    for gg in gs:
                        y.append(l == t)
                        rows["cos"].append(cs)
                        for k in kinds:
                            rows[k].append(inl_value(stats[(qid, gg)], k))
        print(sp, " ".join(f"{k}={roc_auc_score(y, v):.3f}" for k, v in rows.items()),
              f"(n={len(y)}, pos={np.mean(y):.2f})")
    # new-vs-known separation by top1 evidence
    print("\nNew-vs-known AUC of top-1 evidence (higher=known)")
    for sp in ("hyena", "leopard"):
        y, cs1, mx = [], [], {k: [] for k in ("aff", "fund", "ratio")}
        for cands, stats, truth, spv in folds:
            for (qid, cand), t, s in zip(cands, truth, spv):
                if s != sp:
                    continue
                y.append(t != NEW)
                cs1.append(cand[0][1])
                for k in mx:
                    mx[k].append(max(inl_value(stats[(qid, g)], k) for _, _, gs in cand for g in gs))
        print(sp, f"cos1={roc_auc_score(y, cs1):.3f}",
              " ".join(f"max_{k}={roc_auc_score(y, v):.3f}" for k, v in mx.items()))

    # --- grid ---
    def score(kind, w, c, T, pos=2):
        res = {k: [] for k in ("all", "hyena", "leopard", "known", "new")}
        for cands, stats, truth, spv in folds:
            p = rank(cands, stats, kind, w, c, T, pos)
            res["all"].append(map5(p, truth))
            for s in ("hyena", "leopard"):
                m = spv == s
                res[s].append(map5([x for x, k in zip(p, m) if k], truth[m]))
            for n, m in (("known", truth != NEW), ("new", truth == NEW)):
                res[n].append(map5([x for x, k in zip(p, m) if k], truth[m]))
        return {k: float(np.mean(v)) for k, v in res.items()}

    print("\nbaseline-equivalent (w=0):", score("aff", 0, 0, 0.45))
    best = []
    for kind in ("aff", "fund", "ratio", "homo"):
        for w, c in itertools.product([0.02, 0.04, 0.06, 0.1, 0.15], [0, 3, 5, 8]):
            for T in np.arange(0.40, 0.66, 0.025):
                r = score(kind, w, c, T)
                best.append((r["all"], kind, w, c, round(T, 3), r))
    best.sort(key=lambda x: -x[0])
    for b in best[:10]:
        print(b[1:5], {k: round(v, 4) for k, v in b[5].items()})
    for kind in ("aff", "fund", "ratio", "homo"):
        b = next(x for x in best if x[1] == kind)
        print("best", kind, b[1:5], {k: round(v, 4) for k, v in b[5].items()})


if __name__ == "__main__":
    if "--analyse" in sys.argv:
        analyse()
    cfg = dict(kind="aff", w=0.06, c=3, T=0.5)
    for a in sys.argv:
        if a.startswith("--cfg="):
            cfg = eval(a[6:])
    if "--eval" in sys.argv:
        evaluate(make_predict(**cfg))
    if "--submit" in sys.argv:
        t0 = time.time()
        make_submission(make_predict(**cfg), "sift_best")
        print(f"submission {time.time() - t0:.0f}s")
