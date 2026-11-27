import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from cv_v3 import build_or_load_cv_v3, DATA_DIR
from features_v3 import build_v3_feature_bundle, CAND_V3_NAMES, DEFAULT_STAGE1_V3_CFG


def load_v3_raw_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    n_tr, n_te = len(train_df), len(test_df)
    all_identities = np.concatenate([train_df["identity"].values, np.array(["TEST"] * n_te, dtype=object)])
    all_viewpoints = np.concatenate([
        train_df["viewpoint"].fillna("MISSING").values,
        test_df["viewpoint"].fillna("MISSING").values,
    ])
    miew_all = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))
    miew_norms = np.linalg.norm(miew_all, axis=1).astype(np.float32)
    X_norm_all = normalize(miew_all, axis=1).astype(np.float32)
    vlad_all = np.load("cache/vlad_c64_p512.npy")
    fv_all = np.load("cache_v2/fisher_g32_p512.npy")
    offsets = np.load("cache/sift_offsets.npy")
    kp_counts = np.diff(offsets).astype(np.int32)
    sift_z = np.load("cache_v3/sift_v3_peak_pairs_csr.npz")
    sift_csr = (sift_z["row_ptr"], sift_z["col_idx"], sift_z["feat_vals"])
    return train_df, test_df, all_identities, all_viewpoints, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr


def main():
    t0 = time.time()
    train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = load_v3_raw_data()
    folds = build_or_load_cv_v3()
    fd0 = folds[0]
    gi, vi = fd0["gallery_idx"], fd0["val_idx"]

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        g_rows = gi[train_df["species"].values[gi] == sp]
        vm = (train_df["species"].values[vi] == sp)
        v_rows = vi[vm]
        y_v = fd0["val_labels"][vm]
        is_n = fd0["val_is_new"][vm]

        b = build_v3_feature_bundle(
            g_rows, v_rows, sp,
            all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
            sift_csr, top_k_cands=20, stage1_cfg=DEFAULT_STAGE1_V3_CFG[sp]
        )
        Xc = b["X_cand"]
        Xo = b["X_open_sr"]
        assert Xc.shape[2] == len(CAND_V3_NAMES) == 90, f"Cand mismatch: {Xc.shape[2]} vs {len(CAND_V3_NAMES)}"
        assert np.isfinite(Xc).all(), f"NaN/Inf in X_cand ({sp})!"
        assert np.isfinite(Xo).all(), f"NaN/Inf in X_open_sr ({sp})!"

        kn = ~is_n
        c20 = b["cand_names"][kn]
        yk = y_v[kn]
        hit1 = float(np.mean(c20[:, 0] == yk))
        rec12 = float(np.mean([yk[i] in c20[i, :12] for i in range(len(yk))]))
        rec20 = float(np.mean([yk[i] in c20[i] for i in range(len(yk))]))
        print(
            f"[V3 SMOKE PASS | {sp:7s}] time={time.time() - ts:.2f}s | X_cand={Xc.shape}, X_open_sr={Xo.shape} | "
            f"Fold0 Known Top-1={hit1:.4f}, Top-12={rec12:.4f}, Top-20={rec20:.4f} | "
            f"max_mov_inl={b['max_mov_inl'].max():.0f}, max_hom_inl={b['max_hom_inl'].max():.0f}, v_edges={b['V_edge_csr'].nnz//2}"
        )
    print(f"V3 smoke tests passed cleanly in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
