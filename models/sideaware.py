"""Side-aware ranking and new_individual decision (D-013), built on combo_lda_sift.

Known animals in test are often photographed from the flank (left/right) that the gallery
lacks for them (D-012). Their best same-flank similarity is low, so a plain threshold calls them
new. Per species, for each query with a known flank (L or R):

  base(id)  = cos+LDA blend (max over the identity's images) + w * log1p(max(inliers - 3, 0))
              (SIFT bonus only for the combo's top-10 cosine candidates)
  score(id) = base(id) + b * [identity has NO gallery image on the query's flank]

The b-term lifts identities that could only be matched cross-flank. new_individual:
  if top score < T:  new at position `pos_low` (1 = first)
  else:              new at position 2
With b=0, pos_low=1 this is exactly combo_lda_sift. Queries with no flank (front/back/up/down/
missing) use b=0.

Run:  python -m models.sideaware [--submit]     (honest + in-sample tuning on split v2)
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

from models.metric_learning import embed, id_scores
from models.sift_match import compute
from models.sift_rerank import candidates, inl_value
from reid.data import NEW, ROOT, load
from reid.evaluate import make_submission
from reid.metric import map5
from reid.split import N_FOLDS, SIDE, cross_view_mask, make_split

LDA_CFG = dict(lda=128, shrink=0.3)
W_SIFT, C_SIFT = 0.09, 3


def features(gal, Xg, qry, Xq):
    """Per species: dict(qm, uniq, base [nq, nid], lacks [nq, nid] bool, has_side [nq] bool)."""
    cands, pairs = candidates(gal, Xg, qry, Xq)
    stats = compute(pairs)
    gside = gal.viewpoint.map(SIDE).fillna("O").values
    qside = qry.viewpoint.map(SIDE).fillna("O").values
    out = {}
    for s in ("hyena", "leopard"):
        gm = (gal.species == s).values
        qm = np.where((qry.species == s).values)[0]
        Sqg, Sgg = embed(Xg[gm], Xq[qm], gal.identity.values[gm], LDA_CFG)
        uniq, M = id_scores(Sqg, Sgg, gal[gm], qry.iloc[qm], LDA_CFG)
        col = {u: j for j, u in enumerate(uniq)}
        base = M.astype(np.float32).copy()
        for r, q in enumerate(qm):
            qid, cand = cands[q]
            for l, _, gs in cand:
                x = max(inl_value(stats[(qid, g)], "aff") for g in gs)
                base[r, col[l]] += W_SIFT * np.log1p(max(x - C_SIFT, 0))
        # which identities have a gallery image on each flank
        labs = gal.identity.values[gm]
        hasL = np.zeros(len(uniq), bool); hasR = np.zeros(len(uniq), bool)
        for l, sd in zip(labs, gside[gm]):
            if sd == "L": hasL[col[l]] = True
            if sd == "R": hasR[col[l]] = True
        qs = qside[qm]
        lacks = np.zeros((len(qm), len(uniq)), bool)
        lacks[qs == "L"] = ~hasL
        lacks[qs == "R"] = ~hasR
        out[s] = dict(qm=qm, uniq=uniq, base=base, lacks=lacks)
    return out


def rank(feat_s, b, T, pos_low):
    sc = feat_s["base"] + b * feat_s["lacks"]
    o = np.argsort(-sc, 1)[:, :5]
    top1 = sc[np.arange(len(sc)), o[:, 0]]
    preds = []
    for r in range(len(sc)):
        tp = list(feat_s["uniq"][o[r]])
        k = pos_low if top1[r] < T else 2
        preds.append(tp[: k - 1] + [NEW] + tp[k - 1: 4])
    return preds


def make_predict(params):
    def predict(gal, Xg, qry, Xq):
        F = features(gal, Xg, qry, Xq)
        out = [None] * len(qry)
        for s, fs in F.items():
            for q, p in zip(fs["qm"], rank(fs, *params[s])):
                out[q] = p
        return out
    return predict


GRID = list(itertools.product([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                              np.round(np.arange(0.36, 0.60, 0.01), 3), [1, 2]))


def main():
    cache = ROOT / "cache" / "sideaware_feats.pkl"
    train, _, Xtr, _ = load()
    if cache.exists():
        folds = pickle.loads(cache.read_bytes())
    else:
        folds = []
        for f in range(N_FOLDS):
            g, q, t = make_split(f)
            gal = train.iloc[g].reset_index(drop=True)
            qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
            folds.append((features(gal, Xtr[g], qry, Xtr[q]), t, cross_view_mask(f)))
            print("features fold", f, flush=True)
        cache.parent.mkdir(exist_ok=True); cache.write_bytes(pickle.dumps(folds))

    # score table: per fold, species, grid point -> (sum AP, n) on all / xview / known / new
    def fold_scores(F, t, cv, s, prm):
        fs = F[s]; P = rank(fs, *prm); tt = t[fs["qm"]]; c = cv[fs["qm"]]
        ap = np.array([1 / (p.index(x) + 1) if x in p[:5] else 0 for p, x in zip(P, tt)])
        return ap, tt, c
    table = {}
    for fi, (F, t, cv) in enumerate(folds):
        for s in ("hyena", "leopard"):
            for prm in GRID:
                table[fi, s, prm] = fold_scores(F, t, cv, s, prm)

    def total(fis, choice):
        aps, tts, cvs = [], [], []
        for fi in fis:
            for s in ("hyena", "leopard"):
                ap, tt, c = table[fi, s, choice[s]]
                aps.append(ap); tts.append(tt); cvs.append(c)
        ap, tt, c = np.concatenate(aps), np.concatenate(tts), np.concatenate(cvs)
        return dict(all=ap.mean(), known=ap[(tt != NEW) & ~c].mean(), new=ap[tt == NEW].mean(),
                    xview=ap[c].mean())

    def best(fis, allowed):
        ch = {}
        for s in ("hyena", "leopard"):
            ch[s] = max(allowed, key=lambda prm: np.mean([table[fi, s, prm][0].sum() for fi in fis]))
        return ch

    variants = {"combo (b=0, new first if low)": [g for g in GRID if g[0] == 0 and g[2] == 1],
                "+ side bonus b": [g for g in GRID if g[2] == 1],
                "+ new at #2 when low": [g for g in GRID if g[0] == 0],
                "side bonus + pos choice": GRID}
    print("| variant | protocol | all | known | new | xview | params |\n|---|---|---|---|---|---|---|")
    final = None
    for name, allowed in variants.items():
        ch = best(range(N_FOLDS), allowed)
        r = total(range(N_FOLDS), ch)
        print(f"| {name} | in-sample | {r['all']:.4f} | {r['known']:.4f} | {r['new']:.4f} | {r['xview']:.4f} | {ch} |")
        hon = []
        for test_f in range(N_FOLDS):
            tr = [f for f in range(N_FOLDS) if f != test_f]
            hon.append(total([test_f], best(tr, allowed)))
        h = {k: np.mean([x[k] for x in hon]) for k in hon[0]}
        sd = np.std([x["all"] for x in hon], ddof=1)
        print(f"| {name} | honest | {h['all']:.4f} ±{sd:.4f} | {h['known']:.4f} | {h['new']:.4f} | {h['xview']:.4f} | |", flush=True)
        final = ch
    if "--submit" in sys.argv:
        make_submission(make_predict(final), "sideaware")


if __name__ == "__main__":
    main()
