"""new_individual decision via logistic regression P(new), trained by simulating new ids
inside the gallery (leave-one-out = known, leave-identity-out = new). Known-id ranking is the
baseline (max cosine per identity). Final list = candidates sorted by probability
(new: P(new); id j: (1-P(new)) * softmax(s_j / tau)), which maximises expected AP@5.
Optional transductive boost: queries with a near-duplicate query that is itself unmatched.
Run: OMP_NUM_THREADS=4 python -m models.newid_lr [--submit]
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from reid.data import NEW, preprocess
from reid.evaluate import evaluate, make_submission

FLANK = {"left": "L", "frontleft": "L", "backleft": "L", "right": "R", "frontright": "R", "backright": "R"}


def id_scores(S, inv, nid):
    M = np.full((S.shape[0], nid), -1.0, dtype=np.float32)
    np.maximum.at(M.T, inv, S.T)
    return M


def feats(M, vq, best_vps, sizes):
    o = np.sort(M, 1)[:, ::-1]
    t1, t2, t5 = o[:, 0], o[:, 1], o[:, min(4, o.shape[1] - 1)]
    fq = np.array([FLANK.get(v, "O") for v in vq])
    vmatch = np.array([fq[i] != "O" and fq[i] in best_vps[i] for i in range(len(vq))], float)
    vopp = np.array([fq[i] != "O" and fq[i] not in best_vps[i] for i in range(len(vq))], float)
    return np.c_[t1, t1 - t2, t2 - t5, np.log1p(sizes), vmatch, vopp, t1 * vopp]


def species_model(Fg, labs, vg, Fq, vq, tau, use_tr, tr_sim, bias=0.0):
    uniq, inv = np.unique(labs, return_inverse=True)
    nid = len(uniq)
    sizes_id = np.bincount(inv, minlength=nid)
    flank_g = np.array([FLANK.get(v, "O") for v in vg])
    id_flanks = [set(flank_g[inv == j]) for j in range(nid)]
    # ---- simulated training set from the gallery
    S = Fg @ Fg.T
    np.fill_diagonal(S, -1)
    Mk = id_scores(S, inv, nid)                     # leave-one-out (known if own id has >=2)
    own = Mk[np.arange(len(Fg)), inv].copy()
    Mn = Mk.copy(); Mn[np.arange(len(Fg)), inv] = -1  # leave-identity-out (new)
    multi = sizes_id[inv] >= 2
    X, y = [], []
    for M, lab, mask in ((Mk, 0, multi), (Mn, 1, np.ones(len(Fg), bool))):
        M = M[mask]; b = M.argmax(1)
        sz = sizes_id[b] - (b == inv[mask])
        bv = []
        for i, j in zip(np.where(mask)[0], b):
            f = set(flank_g[(inv == j) & (np.arange(len(Fg)) != i)])
            bv.append(f)
        X.append(feats(M, vg[mask], bv, sz)); y.append(np.full(mask.sum(), lab))
    X, y = np.vstack(X), np.concatenate(y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X, y)
    # ---- queries
    Mq = id_scores(Fq @ Fg.T, inv, nid)
    b = Mq.argmax(1)
    Xq = feats(Mq, vq, [id_flanks[j] for j in b], sizes_id[b])
    pnew = 1 / (1 + np.exp(-(clf.decision_function(Xq) + bias)))
    if use_tr and len(Fq) > 1:
        Q = Fq @ Fq.T; np.fill_diagonal(Q, -1)
        # neighbour query that is very similar: share evidence (average P(new) with strong neighbours)
        W = (Q > tr_sim).astype(float)
        deg = W.sum(1)
        pnew = np.where(deg > 0, 0.5 * pnew + 0.5 * (W @ pnew) / np.maximum(deg, 1), pnew)
    order = np.argsort(-Mq, 1)[:, :5]
    out = []
    for r in range(len(Fq)):
        s = Mq[r, order[r]]
        p = np.exp((s - s.max()) / tau); p = p / p.sum() * (1 - pnew[r])
        cands = list(zip(p, uniq[order[r]])) + [(pnew[r], NEW)]
        cands.sort(key=lambda t: -t[0])
        out.append([c[1] for c in cands[:5]])
    return out, pnew


def make_predict(tau=0.05, use_tr=False, tr_sim=0.75, bias=0.0):
    def predict(gal, Xg, qry, Xq):
        Fg, Fq = preprocess(Xg, Xq)
        out = [None] * len(qry)
        for s in ("hyena", "leopard"):
            gm = (gal.species == s).values
            qm = np.where((qry.species == s).values)[0]
            preds, pn = species_model(Fg[gm], gal.identity.values[gm], gal.viewpoint.values[gm],
                                      Fq[qm], qry.viewpoint.values[qm], tau, use_tr, tr_sim, bias)
            for q, p in zip(qm, preds):
                out[q] = p
        return out
    return predict


if __name__ == "__main__":
    for kw in [dict(tau=0.02, bias=-0.5), dict(tau=0.02, bias=-1.0), dict(tau=0.01, bias=-0.5),
               dict(tau=0.03, bias=-1.0), dict(tau=0.02, bias=-1.5)]:
        r = evaluate(make_predict(**kw), verbose=False)
        print(kw, {k: round(v, 4) for k, v in r.items()}, flush=True)
