"""Baseline: cosine k-NN on centred+L2 MiewID, identity score = max similarity.

new_individual goes first if top-1 similarity < T[species], otherwise second (hedge).
Run:  python -m models.baseline_knn [--submit]
"""
import sys

import numpy as np

from reid.data import NEW, preprocess
from reid.evaluate import evaluate, make_submission


def identity_scores(Fg, Fq, labels):
    """Max cosine of each query to each identity. Returns (uniq_labels, scores [nq, nid])."""
    uniq, inv = np.unique(labels, return_inverse=True)
    S = Fq @ Fg.T
    M = np.full((len(Fq), len(uniq)), -np.inf, dtype=np.float32)
    np.maximum.at(M.T, inv, S.T)
    return uniq, M


def make_predict(T={"hyena": 0.45, "leopard": 0.45}, new_pos=2):
    def predict(gal, Xg, qry, Xq):
        Fg, Fq = preprocess(Xg, Xq)
        out = [None] * len(qry)
        for s in ("hyena", "leopard"):
            gm = (gal.species == s).values
            qm = np.where((qry.species == s).values)[0]
            uniq, M = identity_scores(Fg[gm], Fq[qm], gal.identity.values[gm])
            order = np.argsort(-M, axis=1)[:, :5]
            for r, q in enumerate(qm):
                top = list(uniq[order[r]])
                if M[r, order[r, 0]] < T[s]:
                    out[q] = [NEW] + top[:4]
                else:
                    out[q] = top[: new_pos - 1] + [NEW] + top[new_pos - 1 : 4]
        return out
    return predict


if __name__ == "__main__":
    best = None
    for t in np.arange(0.30, 0.66, 0.05):
        print(f"T={t:.2f}")
        r = evaluate(make_predict({"hyena": t, "leopard": t}), verbose=False)
        print(" ", {k: round(v, 4) for k, v in r.items()})
        if best is None or r["all"] > best[1]["all"]:
            best = (t, r)
    print("best T", round(best[0], 2), best[1])
    if "--submit" in sys.argv:
        make_submission(make_predict({"hyena": best[0], "leopard": best[0]}), "baseline_knn")
