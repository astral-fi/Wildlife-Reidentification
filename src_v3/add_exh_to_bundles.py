"""Append exhaustive-SIFT features to v3 bundles (all of them: outer val + sub-folds, test, full seeds).

  python src_v3/add_exh_to_bundles.py [species ...]        # default: leopard
reads  cache_v3/v3_bundles_{sp}.pkl
writes cache_v3/v3_bundles_{sp}_exh.pkl   (originals untouched)
then   V3_BUNDLE_SUFFIX=_exh python src_v3/reproduce_v3_no_optuna.py
       -> submissions_v3/submission_v3_reproduced_no_optuna_exh.csv

Bundle query/gallery rows are re-derived exactly as optuna_stage1_v3.extract_all_v3_bundles does
(same folds and seeds), and checked against each bundle's stored labels before anything is appended.
Pair results go to cache_v3/exh_pairs_leopard.npz (shared with exh_sift_features.py; fold-independent).
Validated on CV in src_v3/ab_exh_stage2.py (leopard +0.0083 MAP@5 at 44% new, 5/5 folds).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, pickle
sys.path.insert(0, "src_v3"); sys.path.insert(0, "src_v2")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize
from cv_v3 import build_or_load_cv_v3, build_open_set_splits, CACHE_V3_DIR
import exh_sift_features as E


def derive_rows(sp, data, train_df, test_df, all_id, X_norm_all):
    """-> list of (bundle_dict, gallery_rows, query_rows) in a fixed order, after label checks."""
    folds = build_or_load_cv_v3()
    n_tr = len(train_df)
    spc = train_df["species"].values
    out = []

    def check(b, q_rows, labels, what):
        assert len(b["cand_names"]) == len(q_rows), f"{what}: {len(b['cand_names'])} bundle rows vs {len(q_rows)} derived"
        if labels is not None and "y_true" in b:
            assert np.array_equal(np.asarray(b["y_true"], dtype=object), np.asarray(labels, dtype=object)), f"{what}: labels differ"

    for od, fd in zip(data["outer_data"], folds):
        fk = fd["fold"]; assert od["fold"] == fk
        gi, vi = fd["gallery_idx"], fd["val_idx"]
        g_rows = gi[spc[gi] == sp]; vm = spc[vi] == sp
        check(od["val_bundle"], vi[vm], fd["val_labels"][vm], f"outer {fk} val")
        out.append((od["val_bundle"], g_rows, vi[vm]))
        subfolds = build_open_set_splits(all_id[g_rows], np.full(len(g_rows), sp, dtype=object), X_norm_all[g_rows],
                                         n_splits=4, enc_thresh=0.80, seed=101 + 37 * fk)
        for sb, sf in zip(od["sub_bundles"], subfolds):
            check(sb, g_rows[sf["val_idx"]], sf["val_labels"], f"outer {fk} sub")
            out.append((sb, g_rows[sf["gallery_idx"]], g_rows[sf["val_idx"]]))
    full_g = np.where(spc == sp)[0].astype(np.int32)
    te_q = n_tr + np.where(test_df["species"].values == sp)[0]
    check(data["test_bundle"], te_q, None, "test")
    out.append((data["test_bundle"], full_g, te_q))
    for sd, sbl in data["full_seed_bundles"].items():
        sf_full = build_open_set_splits(all_id[full_g], np.full(len(full_g), sp, dtype=object), X_norm_all[full_g],
                                        n_splits=5, enc_thresh=0.80, seed=sd)
        for sb, sf in zip(sbl, sf_full):
            check(sb, full_g[sf["val_idx"]], sf["val_labels"], f"seed {sd}")
            out.append((sb, full_g[sf["gallery_idx"]], full_g[sf["val_idx"]]))
    return out


def main(species):
    train_df = pd.read_csv("reid_student_data_main/train.csv"); test_df = pd.read_csv("reid_student_data_main/test.csv")
    all_id = np.concatenate([train_df["identity"].values, np.array(["TEST"] * len(test_df), dtype=object)])
    X = normalize(np.load("reid_student_data_main/miewid_features.npy"))
    for sp in species:
        t0 = time.time()
        src = os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}.pkl")
        data = pickle.load(open(src, "rb"))
        items = derive_rows(sp, data, train_df, test_df, all_id, X)
        print(f"[{sp}] {len(items)} bundles, rows verified", flush=True)
        plans = [E.plan(q, b["cand_names"], g, all_id, X) for b, g, q in items]
        keys = np.concatenate([E.key(np.full(len(gg), qq), gg) for (b, g, q), pq in zip(items, plans)
                               for qq, lst in zip(q, pq) for gg in lst if len(gg)])
        print(f"[{sp}] {len(keys):,} lookups, {len(np.unique(keys)):,} unique pairs", flush=True)
        store = f"cache_v3/exh_pairs_{sp}.npz"
        k_all, F = E.compute_pairs(keys, store)
        for (b, g, q), pq in zip(items, plans):
            cand, open_f = E.summarise(q, pq, k_all, F)
            b["X_cand"] = np.concatenate([b["X_cand"], cand], axis=2).astype(np.float32)
            b["X_open_sr"] = np.c_[b["X_open_sr"], open_f].astype(np.float32)
        dst = os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}_exh.pkl")
        pickle.dump(data, open(dst, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{sp}] wrote {dst} in {time.time() - t0:.0f}s (X_cand +{len(E.EXH_CAND_NAMES)} cols, X_open_sr +{len(E.EXH_OPEN_NAMES)})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["leopard"])
