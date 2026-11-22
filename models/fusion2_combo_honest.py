"""Honest (tune on 2 folds, score 3rd) vs in-sample scoring of models/combo_lda_sift.py.

Run:  OMP_NUM_THREADS=4 python -m models.fusion2_combo_honest
"""
import numpy as np

from models.combo_lda_sift import features, rank
from models.fusion2_rerank import scores
from reid.data import load
from reid.metric import map5
from reid.split import N_FOLDS, make_split

C = 3
W = [0, 0.03, 0.06, 0.09, 0.12]
TS = np.arange(0.40, 0.60, 0.005)


def tune(fs):
    best = None
    for w in W:
        Ts = {}
        for s in ("hyena", "leopard"):
            Ts[s] = max(TS, key=lambda T: np.mean([map5(rank([x for x, sp in zip(F, S) if sp == s], w, C, {s: T}),
                                                          t[S == s]) for F, t, S in fs]))
        v = np.mean([map5(rank(F, w, C, Ts), t) for F, t, _ in fs])
        if best is None or v > best[0]:
            best = (v, w, Ts)
    return best[1], best[2]


if __name__ == "__main__":
    train, _, Xtr, _ = load()
    folds = []
    for f in range(N_FOLDS):
        g, q, t = make_split(f)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        folds.append((features(gal, Xtr[g], qry, Xtr[q]), t, qry.species.values))
    for name, tr in (("in-sample", lambda k: folds),
                     ("honest", lambda k: [folds[j] for j in range(N_FOLDS) if j != k])):
        rr = []
        for k in range(N_FOLDS):
            w, Ts = tune(tr(k))
            F, t, S = folds[k]
            rr.append(scores(rank(F, w, C, Ts), t, S)); print(name, k, w, Ts, rr[-1]["all"], flush=True)
        m = {key: np.mean([r[key] for r in rr]) for key in rr[0]}
        print(f"combo {name}: " + " ".join(f"{a} {v:.4f}" for a, v in m.items()),
              f"±{np.std([r['all'] for r in rr], ddof=1):.4f}", flush=True)
