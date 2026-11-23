"""Rank-level ensemble of b1's V3 with our models (weighted reciprocal-rank fusion).

score(label) = sum_m w_m / (k + rank_m(label)),  rank 1..5, labels missing from a list get 0.
new_individual is just another label. Members per fold come from cache/b1_folds.pkl (b1 V1-V3,
models/b1_eval.py) and cache/sideaware_feats.pkl (our combo = sideaware with b=0, and sideaware).
Tuned honestly (choose on 2 folds, score the 3rd).  Run: python -m models.ens_b1 [--submit]
"""
import itertools
import pickle
import sys

import numpy as np
import pandas as pd

from models.sideaware import rank as side_rank
from reid.data import NEW, ROOT, load
from reid.metric import check_submission, map5
from reid.split import N_FOLDS, cross_view_mask

COMBO = {"hyena": (0, 0.43, 1), "leopard": (0, 0.50, 1)}
SIDE = {"hyena": (0.2, 0.48, 1), "leopard": (0.15, 0.54, 1)}


def ours(F, n, params):
    out = [None] * n
    for s, fs in F.items():
        for q, p in zip(fs["qm"], side_rank(fs, *params[s])):
            out[q] = p
    return out


def rrf(lists, w, k):
    out = []
    for parts in zip(*lists):
        sc = {}
        for wm, p in zip(w, parts):
            for r, l in enumerate(p[:5]):
                sc[l] = sc.get(l, 0) + wm / (k + r + 1)
        out.append([l for l, _ in sorted(sc.items(), key=lambda x: -x[1])][:5])
    return out


def main():
    b1 = pickle.loads((ROOT / "cache" / "b1_folds.pkl").read_bytes())
    feats = pickle.loads((ROOT / "cache" / "sideaware_feats.pkl").read_bytes())
    folds = []
    for f in range(N_FOLDS):
        (p1, p2, p3), t = b1[f]
        F, t2, cv = feats[f]
        assert (t == t2).all()
        folds.append(dict(t=t, cv=cv, members={"b1v3": p3, "b1v2": p2,
                                               "combo": ours(F, len(t), COMBO),
                                               "side": ours(F, len(t), SIDE)}))

    def score(f, names, w, k):
        d = folds[f]; P = rrf([d["members"][n] for n in names], w, k)
        cv = d["cv"]
        return (map5(P, d["t"]), map5([p for p, c in zip(P, cv) if not c], d["t"][~cv]),
                map5([p for p, c in zip(P, cv) if c], d["t"][cv]))

    print("| members | protocol | all (v2) | non-xview | xview | w, k |\n|---|---|---|---|---|---|")
    for name in ("b1v3", "b1v2", "combo", "side"):
        r = np.mean([score(f, [name], [1], 1) for f in range(N_FOLDS)], 0)
        print(f"| {name} alone | — | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} | |")
    best_cfg = None
    for names in (["b1v3", "combo"], ["b1v3", "side"], ["b1v3", "b1v2", "combo"]):
        grid = [(w, k) for w in itertools.product(*[[0.5, 1, 1.5, 2, 3] if i == 0 else [0.25, 0.5, 1]
                                                    for i in range(len(names))])
                for k in (1, 2, 4, 8)]
        tab = {(g, f): score(f, names, list(g[0]), g[1]) for g in grid for f in range(N_FOLDS)}
        ins = max(grid, key=lambda g: np.mean([tab[g, f][0] for f in range(N_FOLDS)]))
        hon = []
        for tf in range(N_FOLDS):
            ch = max(grid, key=lambda g: np.mean([tab[g, f][0] for f in range(N_FOLDS) if f != tf]))
            hon.append(tab[ch, tf])
        hi, sd = np.mean(hon, 0), np.std([h[0] for h in hon], ddof=1)
        ri = np.mean([tab[ins, f] for f in range(N_FOLDS)], 0)
        print(f"| {'+'.join(names)} | in-sample | {ri[0]:.4f} | {ri[1]:.4f} | {ri[2]:.4f} | {ins} |")
        print(f"| {'+'.join(names)} | honest | {hi[0]:.4f} ±{sd:.4f} | {hi[1]:.4f} | {hi[2]:.4f} | |", flush=True)
        if best_cfg is None or hi[0] > best_cfg[0]:
            best_cfg = (hi[0], names, ins)
    print("best", best_cfg)
    if "--submit" in sys.argv:
        _, names, (w, k) = best_cfg
        train, test, _, _ = load()
        sub_dir = ROOT / "submissions"
        src = {"b1v3": "b1/submission_boosting_follow.csv", "b1v2": "b1/submission_svm_kernel.csv",
               "combo": "combo_lda_sift.csv", "side": "sideaware.csv"}
        lists = [pd.read_csv(sub_dir / src[n]).set_index("id").predictions.reindex(test.id).str.split().tolist()
                 for n in names]
        P = rrf(lists, list(w), k)
        sub = pd.DataFrame({"id": test.id, "predictions": [" ".join(p) for p in P]})
        check_submission(sub, test, train)
        sub.to_csv(sub_dir / "ens_b1.csv", index=False)
        print("wrote submissions/ens_b1.csv | new first", f"{np.mean([p[0] == NEW for p in P]):.1%}")


if __name__ == "__main__":
    main()
