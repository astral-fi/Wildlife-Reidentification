"""Evaluate any model on the shared split and write test submissions.

A model is a function

    predict(gallery: pd.DataFrame, Xg: np.ndarray, queries: pd.DataFrame, Xq: np.ndarray)
        -> list[list[str]]

`gallery` has columns id, species, viewpoint, identity; `queries` has id, species, viewpoint
(no identity!). Xg / Xq are the RAW MiewID rows (apply `reid.data.preprocess` yourself).
SIFT for any row: `reid.data.load_sift(row.id)`. Return up to 5 labels per query, best first,
from the gallery identities of the query's species plus `new_individual`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NEW, ROOT, load
from .metric import check_submission, map5
from .split import N_FOLDS, cross_view_mask, make_split


def evaluate(predict, folds=range(N_FOLDS), verbose=True, **split_kw) -> dict:
    """Return dict with mean/per-fold MAP@5 overall, per species, and on known/new queries."""
    train, _, Xtr, _ = load()
    rows = []
    for f in folds:
        t0 = time.time()
        g, q, truth = make_split(f, **split_kw)
        gal = train.iloc[g].reset_index(drop=True)
        qry = train.iloc[q].drop(columns="identity").reset_index(drop=True)
        preds = predict(gal, Xtr[g], qry, Xtr[q])
        assert len(preds) == len(q)
        sp = qry.species.values
        r = dict(fold=f, all=map5(preds, truth), secs=time.time() - t0)
        for s in ("hyena", "leopard"):
            m = sp == s
            r[s] = map5([p for p, k in zip(preds, m) if k], truth[m])
        cv = cross_view_mask(f, **split_kw) if split_kw.get("cross_frac", 1) else np.zeros(len(q), bool)
        for name, m in (("known", (truth != NEW) & ~cv), ("new", truth == NEW), ("xview", cv)):
            r[name] = map5([p for p, k in zip(preds, m) if k], truth[m]) if m.any() else float("nan")
        rows.append(r)
        if verbose:
            print(f"fold {f}: all {r['all']:.4f} | hyena {r['hyena']:.4f} leopard {r['leopard']:.4f}"
                  f" | known {r['known']:.4f} new {r['new']:.4f} xview {r['xview']:.4f} | {r['secs']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = {k: float(df[k].mean()) for k in ("all", "hyena", "leopard", "known", "new", "xview")}
    out["std_all"] = float(df["all"].std())
    if verbose:
        print("mean: " + " | ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)
    return out


def make_submission(predict, name: str) -> Path:
    """Fit on all of train, predict test, check format, write submissions/<name>.csv."""
    train, test, Xtr, Xte = load()
    preds = predict(train, Xtr, test.copy(), Xte)
    sub = pd.DataFrame({"id": test.id, "predictions": [" ".join(p[:5]) for p in preds]})
    check_submission(sub, test, train)
    out = ROOT / "submissions" / f"{name}.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print("wrote", out, "| new_individual first in", f"{np.mean([p[0] == NEW for p in preds]):.1%}")
    return out
