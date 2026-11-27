import os
import sys
sys.path.insert(0, "src_v2")
import numpy as np
import pandas as pd
from cv_v2 import (
    DATA_DIR,
    TEST_SPECIES_WEIGHTS,
    ap_at_5,
    map_at_5_vec,
    build_open_set_splits,
    build_or_load_cv_v2,
)

CACHE_V3_DIR = "cache_v3"
os.makedirs(CACHE_V3_DIR, exist_ok=True)

# Exact 12-estimator 1D/2D KDE MLE validated priors (confirmed by +0.025 LB jump 0.82895 -> 0.854!)
PRIOR_PI_NEW_V3 = {
    "hyena": 0.410,
    "leopard": 0.440,
}


def build_or_load_cv_v3():
    """Reuses the exact leak-free encounter-grouped singleton-preserving 5-fold CV split from V2 for apples-to-apples comparison!"""
    return build_or_load_cv_v2(n_splits=5, enc_thresh=0.80, seed=42, force_rebuild=False)


def eval_cv_v3(folds, train_df, fold_preds_dict, verbose=True, tag="", prior_dict=None):
    priors = prior_dict if prior_dict is not None else PRIOR_PI_NEW_V3
    recs = []
    for fd in folds:
        k = fd["fold"]
        vi = fd["val_idx"]
        vl = fd["val_labels"]
        vn = fd["val_is_new"]
        sps = train_df["species"].values[vi]
        pr = fold_preds_dict[k]
        for i in range(len(vi)):
            recs.append({"sp": sps[i], "is_new": bool(vn[i]), "ap": ap_at_5(vl[i], pr[i])})
    df = pd.DataFrame(recs)
    out = {}
    tot = 0.0
    for sp in ["hyena", "leopard"]:
        sub = df[df["sp"] == sp]
        mk = float(sub[~sub["is_new"]]["ap"].mean())
        mn = float(sub[sub["is_new"]]["ap"].mean())
        pi = priors[sp]
        mp = (1.0 - pi) * mk + pi * mn
        out[f"{sp}_known"] = mk
        out[f"{sp}_new"] = mn
        out[f"{sp}_map5"] = mp
        tot += TEST_SPECIES_WEIGHTS[sp] * mp
    out["lb_proxy"] = float(tot)
    if verbose:
        print(
            f"[{tag:34s}] V3 CV LB-Proxy: {out['lb_proxy']:.5f} | "
            f"Hyena: {out['hyena_map5']:.5f} (kn={out['hyena_known']:.4f}, nw={out['hyena_new']:.4f}) | "
            f"Leopard: {out['leopard_map5']:.5f} (kn={out['leopard_known']:.4f}, nw={out['leopard_new']:.4f})"
        )
    return out
