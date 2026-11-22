"""Learned candidate re-ranker on MiewID cosine k-NN (fusion agent).

For each query, top-K candidate identities from max-cosine; features per (query, candidate);
a classifier trained ONLY on pseudo-queries simulated from the gallery (held-out identities
-> "new", held-out images of kept identities -> known) predicts P(candidate is true id).
new_individual gets score alpha * (1 - sum p) and is ranked among the candidates.
Run: python -m models.fusion_rerank [--submit]
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from reid.data import NEW, preprocess
from reid.evaluate import evaluate, make_submission

K = 20
FLIP = {"left": "right", "right": "left", "frontleft": "frontright", "frontright": "frontleft",
        "backleft": "backright", "backright": "backleft"}


def cand_features(Fg, glab, gview, Fq, qview):
    """Return (feats [nq,K,d], cand labels [nq,K])."""
    uniq, inv = np.unique(glab, return_inverse=True)
    S = Fq @ Fg.T
    nid = len(uniq)
    M = np.full((len(Fq), nid), -2.0, np.float32); np.maximum.at(M.T, inv, S.T)
    cnt = np.bincount(inv, minlength=nid).astype(np.float32)
    Ssum = np.zeros((len(Fq), nid), np.float32); np.add.at(Ssum.T, inv, S.T)
    Mean = Ssum / cnt
    k = min(K, nid)
    order = np.argsort(-M, 1)[:, :k]
    top1 = M[np.arange(len(Fq)), order[:, 0]]
    top2 = M[np.arange(len(Fq)), order[:, 1]] if k > 1 else top1
    # image-level top sims overall
    Ssort = -np.sort(-S, 1)[:, :3]
    feats = np.zeros((len(Fq), k, 16), np.float32)
    members = [np.where(inv == j)[0] for j in range(nid)]
    for i in range(len(Fq)):
        qv = qview[i]; fv = FLIP.get(qv)
        for r, j in enumerate(order[i]):
            m = members[j]; s = S[i, m]
            ss = np.sort(s)[::-1]
            vs = gview[m]
            same = vs == qv; opp = vs == fv if fv else np.zeros(len(m), bool)
            feats[i, r] = (M[i, j], Mean[i, j], ss[1] if len(ss) > 1 else -2, r, top1[i] - M[i, j],
                           np.log1p(cnt[j]), s[same].max() if same.any() else -2, same.mean(),
                           s[opp].max() if opp.any() else -2, opp.mean(), top1[i], top2[i],
                           top1[i] - top2[i], Ssort[i, 0], Ssort[i, 2] if S.shape[1] > 2 else 0,
                           qv == "missing")
    return feats, uniq[order]


def simulate(Fg, glab, gview, rng, new_frac=0.4, known_frac=0.25, reps=2):
    X, y, grp = [], [], []
    ids, inv = np.unique(glab, return_inverse=True)
    for _ in range(reps):
        perm = rng.permutation(len(ids)); sizes = np.bincount(inv)
        n_known = int(known_frac * sizes[sizes >= 2].sum())
        target = n_known * new_frac / (1 - new_frac)
        unseen, tot = set(), 0
        for j in perm:
            if tot >= target: break
            if sizes[j] <= 20: unseen.add(j); tot += sizes[j]
        q, g = [], []
        for j in range(len(ids)):
            m = np.where(inv == j)[0]
            if j in unseen: q += list(m)
            elif len(m) >= 2:
                kk = min(len(m) - 1, max(1, int(round(known_frac * len(m)))))
                p = rng.choice(m, kk, replace=False); q += list(p); g += [x for x in m if x not in set(p)]
            else: g += list(m)
        q, g = np.array(q), np.array(g)
        f, c = cand_features(Fg[g], glab[g], gview[g], Fg[q], gview[q])
        lab = (c == glab[q][:, None])
        X.append(f); y.append(lab)
    return np.concatenate(X), np.concatenate(y)


def make_models(kind):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def rank_out(P, C, alpha, mode):
    out = []
    for p, c in zip(P, C):
        pn = alpha * (1 - p.sum()) if mode == "sum" else alpha * (1 - p.max())
        sc = np.append(p, pn); labs = np.append(c, NEW)
        out.append(list(labs[np.argsort(-sc)[:5]]))
    return out


CACHE = {}


def make_predict(kind="hgb", alpha=1.0, mode="sum", reps=2, seed=0):
    def predict(gal, Xg, qry, Xq):
        key = (kind, reps, seed, len(gal), len(qry), float(Xq[:3].sum()))
        if key not in CACHE:
            Fg, Fq = preprocess(Xg, Xq)
            res = [None] * len(qry)
            for s in ("hyena", "leopard"):
                gm = (gal.species == s).values; qm = np.where((qry.species == s).values)[0]
                glab = np.asarray(gal.identity.values[gm], dtype=object); gview = np.asarray(gal.viewpoint.values[gm], dtype=object)
                Xs, ys = simulate(Fg[gm], glab, gview, np.random.default_rng(seed), reps=reps)
                clf = make_models(kind).fit(Xs.reshape(-1, Xs.shape[-1]), ys.ravel())
                f, c = cand_features(Fg[gm], glab, gview, Fq[qm], np.asarray(qry.viewpoint.values[qm], dtype=object))
                P = clf.predict_proba(f.reshape(-1, f.shape[-1]))[:, 1].reshape(f.shape[:2])
                for r, q in enumerate(qm): res[q] = (P[r], c[r])
            CACHE[key] = res
        res = CACHE[key]
        return rank_out([r[0] for r in res], [r[1] for r in res], alpha, mode)
    return predict


if __name__ == "__main__":
    results = []
    for kind in ("lr", "hgb"):
        for mode in ("sum", "max"):
            for a in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                r = evaluate(make_predict(kind, a, mode), verbose=False)
                results.append((kind, mode, a, r))
                print(kind, mode, a, {k: round(v, 4) for k, v in r.items()}, flush=True)
    best = max(results, key=lambda t: t[3]["all"])
    print("BEST", best)
    if "--submit" in sys.argv:
        make_submission(make_predict(best[0], best[2], best[1]), "fusion_best")
