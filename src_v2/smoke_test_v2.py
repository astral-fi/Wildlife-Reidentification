import os
import time
import numpy as np
import pandas as pd

from cv_v2 import build_or_load_cv_v2, DATA_DIR
from features_v2 import build_v2_feature_bundle, CAND_V2_NAMES, DEFAULT_STAGE1_CFG
from optuna_stage1_and_extract import load_all_raw_data


def main():
    t0 = time.time()
    train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = load_all_raw_data()
    folds = build_or_load_cv_v2()
    fd0 = folds[0]
    gi, vi = fd0["gallery_idx"], fd0["val_idx"]

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        g_rows = gi[train_df["species"].values[gi] == sp]
        vm = (train_df["species"].values[vi] == sp)
        v_rows = vi[vm]
        y_v = fd0["val_labels"][vm]
        is_n = fd0["val_is_new"][vm]

        b = build_v2_feature_bundle(
            g_rows, v_rows, sp,
            all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
            sift_csr, top_k_cands=12, stage1_cfg=DEFAULT_STAGE1_CFG[sp]
        )
        Xc = b["X_cand"]
        Xo = b["X_open_base"]
        assert Xc.shape[2] == len(CAND_V2_NAMES) == 67, f"Cand feat mismatch: {Xc.shape[2]} vs {len(CAND_V2_NAMES)}"
        assert np.isfinite(Xc).all(), f"NaN/Inf in X_cand ({sp})!"
        assert np.isfinite(Xo).all(), f"NaN/Inf in X_open_base ({sp})!"

        kn = ~is_n
        rec12 = float(np.mean([y_v[kn][i] in b["cand_names"][kn][i] for i in range(int(kn.sum()))]))
        hit1 = float(np.mean(b["cand_names"][kn, 0] == y_v[kn]))
        print(
            f"[SMOKE PASS | {sp:7s}] time={time.time() - ts:.2f}s | X_cand={Xc.shape}, X_open={Xo.shape} | "
            f"Fold0 Known Top-1={hit1:.4f}, Top-12 Recall={rec12:.4f} | max_sim_inl={b['max_sim_inl'].max():.0f}"
        )

    print(f"All smoke tests passed cleanly in {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
