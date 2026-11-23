"""Run b1's pipeline on our shared split (v2 folds; v1 queries are a subset) and cache the
ranked predictions of V1/V2/V3 per fold in cache/b1_folds.pkl.  Run: python -m models.b1_eval"""
import pickle, tempfile
from pathlib import Path
import numpy as np
from models import b1_pipeline as b
from reid.data import ROOT, load
from reid.evaluate import map5
from reid.split import N_FOLDS, cross_view_mask, make_split

import os
TAG = os.environ.get("B1_TAG", "")
OUT = ROOT / "cache" / f"b1_folds{TAG}.pkl"

def run_fold(data, store, f):
    g, q, t = make_split(f)
    with tempfile.TemporaryDirectory() as d:
        ep, res = b.fit_and_predict_versions(data, np.sort(g), q, store, Path(d))
    pos = {r: i for i, r in enumerate(ep.rows)}
    order = [pos[x] for x in q]
    return [[list(p[i])[:5] for i in order] for p in res], t

if __name__ == "__main__":
    data = b.r.Data(b.DATA_ROOT); store = b.r.SiftStore(data, 6)
    done = pickle.loads(OUT.read_bytes()) if OUT.exists() else {}
    for f in range(N_FOLDS):
        if f not in done:
            done[f] = run_fold(data, store, f); OUT.write_bytes(pickle.dumps(done))
        preds, t = done[f]; cv = cross_view_mask(f)
        for v, P in enumerate(preds, 1):
            print(f"fold {f} V{v}: all {map5(P, t):.4f} | non-xview {map5([p for p, c in zip(P, cv) if not c], t[~cv]):.4f} | xview {map5([p for p, c in zip(P, cv) if c], t[cv]):.4f}", flush=True)
