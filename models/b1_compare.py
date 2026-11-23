"""Compare b1 pipeline variants cached by models/b1_eval.py (cache/b1_folds<TAG>.pkl).

Reports V3 MAP@5 on split v2: all, non-cross-view (the trusted proxy, D-015) and cross-view,
per fold and mean.  Run: python -m models.b1_compare "" _refs4 ...
"""
import pickle
import sys

import numpy as np

from reid.data import ROOT
from reid.metric import map5
from reid.split import cross_view_mask

for tag in sys.argv[1:] or [""]:
    d = pickle.loads((ROOT / "cache" / f"b1_folds{tag}.pkl").read_bytes())
    rows = []
    for f in sorted(d):
        P, t = d[f][0][2], d[f][1]
        cv = cross_view_mask(f)
        rows.append((map5(P, t), map5([p for p, c in zip(P, cv) if not c], t[~cv]),
                     map5([p for p, c in zip(P, cv) if c], t[cv])))
    r = np.array(rows)
    print(f"{tag or 'b1 original':12s} folds {len(r)} | all {r[:, 0].mean():.4f} | non-xview "
          f"{r[:, 1].mean():.4f} (per fold {np.round(r[:, 1], 4).tolist()}) | xview {r[:, 2].mean():.4f}")
