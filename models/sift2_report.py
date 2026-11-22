"""Read cache/sift2_grid.pkl: in-sample best per species, honest leave-one-fold-out score,
and ablations. Run: python -m models.sift2_report"""
import pickle
import numpy as np
from models.sift2_tune import best_TH, apply
from models.sift2_feats import FEATS

res = pickle.load(open(FEATS.with_name("sift2_grid.pkl"), "rb"))
COMBO = ("cos", 10, 2, 0, "max", 0)


def pick(sp, folds, filt=lambda k: True, hs=None):
    b = (-1,)
    for k, arrs in res.items():
        if k[0] != sp or not filt(k):
            continue
        m, T, H = best_TH([arrs[f] for f in folds], hs)
        if m > b[0]:
            b = (m, k, T, H)
    return b


def honest(filt=lambda k: True, verbose=False, hs=None):
    per = []
    for f in range(3):
        tr = [g for g in range(3) if g != f]
        aps = []
        for sp in ("hyena", "leopard"):
            m, k, T, H = pick(sp, tr, filt, hs)
            aps.append(apply(res[k][f], T, H))
            if verbose:
                print(f"  fold {f} {sp}: {k[1:]} T={T:.3f} H={H}")
        per.append(np.concatenate(aps).mean())
    return np.mean(per), np.std(per, ddof=1)


def insample(filt=lambda k: True, hs=None):
    tot, out = [], {}
    for sp in ("hyena", "leopard"):
        m, k, T, H = pick(sp, range(3), filt, hs)
        out[sp] = (round(m, 4), k[1:], round(T, 3), H)
        tot.append([apply(res[k][f], T, H) for f in range(3)])
    allv = [np.concatenate([tot[0][f], tot[1][f]]).mean() for f in range(3)]
    return np.mean(allv), out


for name, filt, hs in [("combo struct, w/c/T per species", lambda k: k[1] == COMBO and k[3] == 3 and k[2] == 0.09, [np.inf]),
                   ("combo struct, tuned w,c per species, H=inf", lambda k: k[1] == COMBO, [np.inf]),
                   ("combo struct + H rule", lambda k: k[1] == COMBO, None),
                   ("all structs, H=inf", lambda k: True, [np.inf]),
                   ("all structs", lambda k: True, None),
                   ("all structs, viewpoint off", lambda k: k[1][3] == 0, None),
                   ("all structs, P<=2", lambda k: k[1][2] <= 2, None),
                   ("all structs, raw counts", lambda k: k[1][5] == 0, None)]:
    ins = insample(filt, hs)
    h = honest(filt, hs=hs)
    print(f"{name}: in-sample {ins[0]:.4f} | honest {h[0]:.4f} ±{h[1]:.4f}\n   {ins[1]}", flush=True)
print("honest details, all structs:"); honest(verbose=True)
