"""Exhaustive-SIFT candidate features for v3 bundles (leopard).

For each query and each of its top-K candidate identities, SIFT-verify the query against up to CAP
gallery images of that identity (most MiewID-similar first) and summarise per identity:
  exh_mov_max, exh_mov_top2, exh_hom_max, exh_fund_max   (log1p), plus per-query best / margin.
Pair results are cached in cache_v3/exh_pairs_leopard.npz (a (q, g) result does not depend on the fold).

  python src_v3/exh_sift_features.py   # -> cache_v3/exh_feats_{cv,test}_leopard.pkl (pairs cached in exh_pairs_leopard.npz)

`add_exh_features(bundle, feats)` appends them to a bundle's X_cand / open features (see bottom).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, time, pickle
sys.path.insert(0, "src_v3"); sys.path.insert(0, "src_v2")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize
import build_sift_v3_and_fv as V

CAP = 24
PAIRS = "cache_v3/exh_pairs_leopard.npz"
COLS = [5, 9, 10]          # moving-sim inliers, homography inliers, fundamental inliers
EXH_CAND_NAMES = ["exh_mov_max", "exh_mov_top2", "exh_hom_max", "exh_fund_max", "exh_n_checked",
                  "exh_mov_minus_best", "exh_is_best"]
EXH_OPEN_NAMES = ["exh_best", "exh_margin", "exh_best_is_top_cand"]


N_ALL = 9954


def key(a, b):
    a, b = np.minimum(a, b), np.maximum(a, b)
    return a.astype(np.int64) * N_ALL + b.astype(np.int64)


def plan(queries, cand_lists, gallery_rows, ids_all, X):
    """-> per query: list (per candidate) of gallery-row arrays (<= CAP, most MiewID-similar first)."""
    gal_ids = ids_all[gallery_rows]
    by_id = {c: gallery_rows[gal_ids == c] for c in np.unique(gal_ids)}
    return [[by_id[c][np.argsort(-(X[by_id[c]] @ X[q]))][:CAP] for c in cands]
            for q, cands in zip(queries, cand_lists)]


def compute_pairs(all_keys, store):
    have = np.load(store) if os.path.exists(store) else None
    keys = np.unique(all_keys)
    if have is not None:
        keys = np.setdiff1d(keys, have["keys"])
    if len(keys):
        J = np.column_stack([keys // N_ALL, keys % N_ALL]).astype(np.int32)
        t0 = time.time()
        with ProcessPoolExecutor(16, initializer=V._init_worker) as ex:
            F = np.vstack(list(ex.map(V._compute_v3_pair_batch, np.array_split(J, max(16, len(J) // 4000)))))
        F = F[:, COLS].astype(np.float32)
        print(f"  computed {len(J):,} pairs in {time.time() - t0:.0f}s", flush=True)
        if have is not None:
            keys = np.r_[have["keys"], keys]; F = np.r_[have["feats"], F]
        o = np.argsort(keys); keys, F = keys[o], F[o]
        np.savez(store, keys=keys, feats=F)
    z = np.load(store)
    return z["keys"], z["feats"]


def summarise(queries, per_q, keys, F):
    n, K = len(queries), len(per_q[0])
    out = np.zeros((n, K, len(EXH_CAND_NAMES)), np.float32)
    for i, (q, lst) in enumerate(zip(queries, per_q)):
        for k, g in enumerate(lst):
            if len(g) == 0:
                continue
            f = F[np.searchsorted(keys, key(np.full(len(g), q), g))]
            mv = np.sort(f[:, 0])[::-1]
            out[i, k, :5] = [np.log1p(mv[0]), np.log1p(mv[:2].sum()), np.log1p(f[:, 1].max()),
                             np.log1p(f[:, 2].max()), np.log1p(len(g))]
        best = out[i, :, 0].max()
        out[i, :, 5] = out[i, :, 0] - best
        out[i, :, 6] = out[i, :, 0] >= best
    srt = -np.sort(-out[:, :, 0], axis=1)
    open_f = np.c_[srt[:, 0], srt[:, 0] - srt[:, 1], out[:, 0, 6]]   # cand 0 = stage-1 top candidate
    return out, open_f.astype(np.float32)


def build_test_bundle(tr, te):
    path = "cache_v3/v3_test_default_leopard.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from smoke_test_v3 import load_v3_raw_data
    from features_v3 import build_v3_feature_bundle, DEFAULT_STAGE1_V3_CFG
    _, _, all_id, all_vp, miew_all, miew_norms, _, vlad_all, fv_all, kp_counts, sift_csr = load_v3_raw_data()
    g = np.where(tr.species.values == "leopard")[0]
    q = len(tr) + np.where(te.species.values == "leopard")[0]
    b = build_v3_feature_bundle(g, q, "leopard", all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all,
                                kp_counts, sift_csr, top_k_cands=20, stage1_cfg=DEFAULT_STAGE1_V3_CFG["leopard"])
    keep = {k: b[k] for k in ["cand_names", "stage1_top_scores", "X_cand", "X_open_sr", "eff_vp_q"]}
    keep.update(g_rows=g, q_rows=q)
    pickle.dump(keep, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return keep


def run():
    tr = pd.read_csv("reid_student_data_main/train.csv"); te = pd.read_csv("reid_student_data_main/test.csv")
    ids_all = np.asarray(list(tr.identity.values) + ["TEST"] * len(te), dtype=object)
    X = normalize(np.load("reid_student_data_main/miewid_features.npy"))
    B = pickle.load(open("cache_v3/v3_outer_default_leopard.pkl", "rb"))
    tb = build_test_bundle(tr, te)
    parts = [(b["v_rows"], b["cand_names"], b["g_rows"]) for b in B] + [(tb["q_rows"], tb["cand_names"], tb["g_rows"])]
    plans = [plan(qs, cands, g, ids_all, X) for qs, cands, g in parts]
    all_keys = np.concatenate([key(np.full(len(gg), q), gg) for (qs, _, _), pq in zip(parts, plans)
                               for q, lst in zip(qs, pq) for gg in lst if len(gg)])
    print(f"{len(all_keys):,} pair lookups, {len(np.unique(all_keys)):,} unique pairs", flush=True)
    keys, F = compute_pairs(all_keys, PAIRS)
    res = [summarise(qs, pq, keys, F) for (qs, _, _), pq in zip(parts, plans)]
    pickle.dump(res[:-1], open("cache_v3/exh_feats_cv_leopard.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(res[-1], open("cache_v3/exh_feats_test_leopard.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("saved exhaustive-SIFT features for CV folds and test")


def add_exh_features(bundle, feats):
    """Return copies of (X_cand, X_open) with the exhaustive-SIFT features appended."""
    cand, open_f = feats
    return np.concatenate([bundle["X_cand"], cand], axis=2), np.c_[bundle["X_open_sr"], open_f]


if __name__ == "__main__":
    run()
