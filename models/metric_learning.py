"""Metric learning / re-ranking variants on MiewID (agent: metric). Fits only on the gallery.

Run: OMP_NUM_THREADS=4 python -m models.metric_learning [--submit]
Each variant -> per-species identity score matrix; the new_individual threshold is swept
per species offline (new first if top1 < T, else new at #2).
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from reid.data import NEW, load, l2n
from reid.split import make_split, N_FOLDS
from reid.metric import map5
from reid.evaluate import make_submission

FLIP = {"left": "right", "right": "left", "frontleft": "frontright", "frontright": "frontleft",
        "backleft": "backright", "backright": "backleft"}


def embed(Xg, Xq, yg, cfg):
    mu = Xg.mean(0)
    Fg, Fq = l2n(Xg - mu), l2n(Xq - mu)
    sims = []
    if cfg.get("w_cos", 1.0) > 0:
        sims.append((cfg.get("w_cos", 1.0), Fg, Fq))
    if cfg.get("lda"):
        k = min(cfg["lda"], len(np.unique(yg)) - 1)
        p = PCA(cfg.get("pca", 256), random_state=0).fit(Fg)
        Pg, Pq = p.transform(Fg), p.transform(Fq)
        lda = LDA(solver="eigen", shrinkage=cfg.get("shrink", 0.1), n_components=k).fit(Pg, yg)
        Lg, Lq = lda.transform(Pg), lda.transform(Pq)
        m = Lg.mean(0)
        sims.append((cfg.get("w_lda", 1.0), l2n(Lg - m), l2n(Lq - m)))
    if cfg.get("white"):
        p = PCA(cfg["white"], whiten=True, random_state=0).fit(Fg)
        sims.append((cfg.get("w_white", 1.0), l2n(p.transform(Fg)), l2n(p.transform(Fq))))
    W = sum(w for w, _, _ in sims)
    Sgg = sum(w * A @ A.T for w, A, _ in sims) / W
    Sqg = sum(w * B @ A.T for w, A, B in sims) / W
    return Sqg, Sgg


def id_scores(Sqg, Sgg, gal_s, qry_s, cfg):
    yg = gal_s.identity.values
    # alpha-QE on queries: average query sim with sims of its top-k gallery neighbours
    if cfg.get("qe"):
        k, a = cfg["qe"], cfg.get("qe_a", 3.0)
        nn = np.argsort(-Sqg, 1)[:, :k]
        w = np.clip(np.take_along_axis(Sqg, nn, 1), 0, None) ** a
        Sqg = (Sqg + np.einsum("qk,qkg->qg", w, Sgg[nn])) / (1 + w.sum(1, keepdims=True))
    if cfg.get("vp"):
        vq = np.asarray(qry_s.viewpoint, dtype=object)[:, None]; vg = np.asarray(gal_s.viewpoint, dtype=object)[None, :]
        flip = np.vectorize(lambda a, b: FLIP.get(a) == b)(vq, vg)
        Sqg = Sqg + cfg["vp"] * (vq == vg) - cfg.get("vp_flip", 0) * flip
    uniq, inv = np.unique(yg, return_inverse=True)
    agg = cfg.get("agg", "max")
    M = np.full((len(Sqg), len(uniq)), -np.inf, dtype=np.float64)
    np.maximum.at(M.T, inv, Sqg.T)
    if agg != "max":
        # top-2 mean mixed with max
        M2 = np.full_like(M, -np.inf)
        for j in range(len(uniq)):
            c = Sqg[:, inv == j]
            if c.shape[1] >= 2:
                M2[:, j] = np.sort(c, 1)[:, -2]
            else:
                M2[:, j] = c[:, 0]
        b = cfg.get("agg_b", 0.3)
        M = (1 - b) * M + b * M2
    return uniq, M


def scores_all(gal, Xg, qry, Xq, cfg):
    res = {}
    for s in ("hyena", "leopard"):
        gm = (gal.species == s).values
        qm = np.where((qry.species == s).values)[0]
        Sqg, Sgg = embed(Xg[gm], Xq[qm], gal.identity.values[gm], cfg)
        uniq, M = id_scores(Sqg, Sgg, gal[gm], qry.iloc[qm], cfg)
        o = np.argsort(-M, 1)[:, :5]
        res[s] = (qm, uniq[o], M[np.arange(len(M)), o[:, 0]])
    return res


def build(res, n, T):
    out = [None] * n
    for s, (qm, top, t1) in res.items():
        for r, q in enumerate(qm):
            tp = list(top[r])
            out[q] = [NEW] + tp[:4] if t1[r] < T[s] else tp[:1] + [NEW] + tp[1:4]
    return out


def run(cfg):
    train, _, Xtr, _ = load()
    folds = []
    for f in range(N_FOLDS):
        g, q, truth = make_split(f)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].reset_index(drop=True)
        folds.append((scores_all(gal, Xtr[g], qry, Xtr[q], cfg), truth, qry.species.values))
    best = {}
    for s in ("hyena", "leopard"):
        allt = np.concatenate([fo[0][s][2] for fo in folds])
        grid = np.quantile(allt, np.linspace(0.05, 0.95, 91))
        def sc(t):
            v = []
            for res, truth, sp in folds:
                qm, top, t1 = res[s]
                p = build({s: res[s]}, len(truth), {s: t})
                v.append(map5([p[i] for i in qm], truth[qm]))
            return np.mean(v)
        best[s] = max(grid, key=sc)
    rows = []
    for res, truth, sp in folds:
        p = build(res, len(truth), best)
        r = dict(all=map5(p, truth))
        for s in ("hyena", "leopard"):
            m = np.where(sp == s)[0]; r[s] = map5([p[i] for i in m], truth[m])
        for nm, m in (("known", truth != NEW), ("new", truth == NEW)):
            m = np.where(m)[0]; r[nm] = map5([p[i] for i in m], truth[m])
        rows.append(r)
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out, best


def make_predict(cfg, T):
    def predict(gal, Xg, qry, Xq):
        return build(scores_all(gal, Xg, qry, Xq, cfg), len(qry), T)
    return predict


VARIANTS = {
    "base": {},
    "lda64": dict(lda=64, w_cos=0),
    "lda128": dict(lda=128, w_cos=0),
    "lda128_s0.3": dict(lda=128, shrink=0.3, w_cos=0),
    "cos+lda128": dict(lda=128),
    "cos+lda128_s0.3": dict(lda=128, shrink=0.3),
    "white256": dict(white=256, w_cos=0),
    "cos+white256": dict(white=256),
    "agg_top2mix": dict(agg="mix"),
    "qe3": dict(qe=3),
    "qe5": dict(qe=5),
    "vp0.05": dict(vp=0.05),
    "vp0.05_flip0.05": dict(vp=0.05, vp_flip=0.05),
    "cl_s0.5": dict(lda=128, shrink=0.5),
    "cl200_s0.3": dict(lda=200, shrink=0.3),
    "cl128_s0.3_wl0.5": dict(lda=128, shrink=0.3, w_lda=0.5),
    "cl128_s0.3_wl2": dict(lda=128, shrink=0.3, w_lda=2.0),
    "cl_s0.3+vp": dict(lda=128, shrink=0.3, vp=0.05),
    "cl_s0.3+vpflip": dict(lda=128, shrink=0.3, vp=0.05, vp_flip=0.05),
}

if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(VARIANTS)
    for n in names:
        r, T = run(VARIANTS[n])
        print(f"{n:22s} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()),
              "T", {k: round(float(v), 3) for k, v in T.items()}, flush=True)
