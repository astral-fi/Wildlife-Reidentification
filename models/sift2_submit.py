"""Final sift2 config (chosen in-sample on 3 folds; honest 2-tune/1-score 0.9263 vs combo 0.9227).
Run: python -m models.sift2_submit [--eval] [--submit]"""
import sys
import time

import numpy as np

from models.sift2_feats import features
from reid.data import NEW
from reid.evaluate import evaluate, make_submission

# struct = (candidate mode, K ids, P images/id, viewpoint-first, aggregation, kp-normalised)
CFG = {"hyena": (("blend", 15, 3, 1, "max", 0), 0.06, 3, 0.48),
       "leopard": (("cos", 15, 5, 0, "sum2", 0), 0.09, 6, 0.55)}


def rank(F):
    preds = []
    for s, qvp, cand in F:
        (mode, k, p, vp, agg, norm), w, c, T = CFG[s]
        key = (lambda x: -x[1]) if mode == "blend" else (lambda x: -x[2])
        sc = []
        for l, b, _, ims in sorted(cand, key=key)[:k]:
            if vp and isinstance(qvp, str):
                ims = sorted(ims, key=lambda m: (isinstance(m[4], str) and m[4] != qvp, -m[0]))
            v = sorted((a / np.sqrt(max(min(na, nb), 1)) * 10 if norm else a
                        for _, a, na, nb, _ in ims[:p]), reverse=True)
            x = v[0] + (v[1] if agg == "sum2" and len(v) > 1 else 0)
            sc.append((b + w * np.log1p(max(x - c, 0)), l))
        sc.sort(reverse=True)
        top = [l for _, l in sc]
        preds.append([NEW] + top[:4] if sc[0][0] < T else top[:1] + [NEW] + top[1:4])
    return preds


def predict(gal, Xg, qry, Xq):
    t0 = time.time()
    F, n = features(gal, Xg, qry, Xq)
    print(f"{len(qry)} queries, {n} pairs, {time.time() - t0:.0f}s", flush=True)
    return rank(F)


if __name__ == "__main__":
    if "--eval" in sys.argv:
        evaluate(predict)
    if "--submit" in sys.argv:
        make_submission(predict, "sift2_best")
