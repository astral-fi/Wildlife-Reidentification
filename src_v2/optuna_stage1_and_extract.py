import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import normalize

from cv_v2 import build_or_load_cv_v2, build_open_set_splits, DATA_DIR, CACHE_V2_DIR
from features_v2 import build_v2_feature_bundle, DEFAULT_STAGE1_CFG

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_all_raw_data():
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

    sift_z = np.load("cache_v2/sift_peak_pairs_csr.npz")
    sift_csr = (sift_z["row_ptr"], sift_z["col_idx"], sift_z["feat_vals"])
    return train_df, test_df, all_identities, all_viewpoints, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr


def _eval_stage1_mats(u_cls, mats, y_kn, kn_mask, w_dict):
    C = sum(w_dict[k] * mats[k][kn_mask] for k in w_dict)
    top12 = np.argsort(-C, axis=1)[:, :12]
    c12 = u_cls[top12]
    hits12 = float(np.mean([y_kn[i] in c12[i] for i in range(len(y_kn))]))
    aps = np.zeros(len(y_kn), dtype=np.float64)
    for r in range(5):
        m = (c12[:, r] == y_kn) & (aps == 0.0)
        aps[m] = 1.0 / (r + 1)
    map5 = float(aps.mean())
    return 0.45 * hits12 + 0.55 * map5, hits12, map5


def tune_stage1_optuna(folds, train_df, all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts, sift_csr, n_geom_trials=14, n_weight_trials=90):
    """
    Two-Phase Decoupled Optuna Stage A:
      Phase A1 (8-dim Subspace Geometry, 14 TPE trials/species on Folds 0-1):
        Tunes `spow`, `p_mid`, `reg_mid`, `d_mid`, `lda_shrink`, `k1_zhong`, `k2_zhong`, `diff_alpha`.
      Phase A2 (12-dim Stage-1 Fusion Weights, 90 TPE trials/species across ALL 5 outer folds in <15s!):
        Precomputes the 12 similarity matrices once on Folds 0-4, then runs 90 TPE trials optimizing
        `0.45 * Top-12 Recall + 0.55 * Known MAP@5` across all 5 folds!
    """
    w_keys = ["w_p60", "w_sub", "w_lda", "w_wccn", "w_kiss", "w_cca", "w_soft", "w_zhong", "w_ppr", "w_fv", "w_vl", "w_sift"]
    best_cfgs = {}
    for sp in ["hyena", "leopard"]:
        print(f"\n=== OPTUNA STAGE A1 (Subspace Geometry, {n_geom_trials} trials) [{sp.upper()}] ===")
        t_sp = time.time()
        base_w = {k: DEFAULT_STAGE1_CFG[sp][k] for k in w_keys}

        def obj_geom(trial):
            cfg = dict(DEFAULT_STAGE1_CFG[sp])
            cfg["spow"] = trial.suggest_float("spow", 0.72, 0.98)
            cfg["p_mid"] = trial.suggest_float("p_mid", 0.48, 0.68)
            cfg["reg_mid"] = trial.suggest_float("reg_mid", 8e-4, 4e-3, log=True)
            cfg["d_mid"] = trial.suggest_int("d_mid", 512, 768, step=64) if sp == "hyena" else trial.suggest_int("d_mid", 640, 1024, step=64)
            cfg["lda_shrink"] = trial.suggest_float("lda_shrink", 0.12, 0.35)
            cfg["k1_zhong"] = trial.suggest_int("k1_zhong", 12, 24, step=2)
            cfg["k2_zhong"] = trial.suggest_int("k2_zhong", 4, 8)
            cfg["diff_alpha"] = trial.suggest_float("diff_alpha", 0.70, 0.90)

            scs = []
            for fk in [0, 1]:
                fd = folds[fk]
                gi, vi = fd["gallery_idx"], fd["val_idx"]
                g_rows = gi[train_df["species"].values[gi] == sp]
                vm = (train_df["species"].values[vi] == sp)
                v_rows = vi[vm]
                y_v = fd["val_labels"][vm]
                kn = ~fd["val_is_new"][vm]

                res = build_v2_feature_bundle(
                    g_rows, v_rows, sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                    sift_csr, top_k_cands=12, stage1_cfg=cfg, return_stage1_mats=True
                )
                obj_v, _, _ = _eval_stage1_mats(res["u_cls"], res["mats"], y_v[kn], kn, base_w)
                scs.append(obj_v)
            return float(np.mean(scs))

        st_g = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        st_g.enqueue_trial({k: DEFAULT_STAGE1_CFG[sp][k] for k in ["spow", "p_mid", "reg_mid", "d_mid", "lda_shrink", "k1_zhong", "k2_zhong", "diff_alpha"]})
        st_g.optimize(obj_geom, n_trials=n_geom_trials, show_progress_bar=False)

        best_c = dict(DEFAULT_STAGE1_CFG[sp])
        best_c.update(st_g.best_params)
        print(f"[{sp.upper()}] Stage A1 Best Geom Obj={st_g.best_value:.5f} (trial #{st_g.best_trial.number}, baseline={st_g.trials[0].value:.5f}) in {time.time() - t_sp:.1f}s")

        # Phase A2: Precompute the 12 similarity matrices across ALL 5 outer folds ONCE, then run 90 TPE trials in <15s!
        print(f"=== OPTUNA STAGE A2 (5-Fold Retrieval Fusion Weights, {n_weight_trials} trials) [{sp.upper()}] ===")
        tw = time.time()
        fold_mats = []
        for fk in range(5):
            fd = folds[fk]
            gi, vi = fd["gallery_idx"], fd["val_idx"]
            g_rows = gi[train_df["species"].values[gi] == sp]
            vm = (train_df["species"].values[vi] == sp)
            v_rows = vi[vm]
            y_v = fd["val_labels"][vm]
            kn = ~fd["val_is_new"][vm]
            res = build_v2_feature_bundle(
                g_rows, v_rows, sp,
                all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                sift_csr, top_k_cands=12, stage1_cfg=best_c, return_stage1_mats=True
            )
            fold_mats.append((res["u_cls"], res["mats"], y_v[kn], kn))

        base_evals = [_eval_stage1_mats(u, m, yk, kn, base_w) for (u, m, yk, kn) in fold_mats]
        print(f"  [{sp}] 5-Fold Baseline Stage-1: Top-12 Rec={np.mean([e[1] for e in base_evals]):.5f}, Known MAP@5={np.mean([e[2] for e in base_evals]):.5f}")

        def obj_weights(trial):
            wd = {
                "w_p60": trial.suggest_float("w_p60", 0.10, 0.40),
                "w_sub": trial.suggest_float("w_sub", 0.02, 0.22),
                "w_lda": trial.suggest_float("w_lda", 0.05, 0.30),
                "w_wccn": trial.suggest_float("w_wccn", 0.00, 0.16),
                "w_kiss": trial.suggest_float("w_kiss", 0.00, 0.18),
                "w_cca": trial.suggest_float("w_cca", 0.00, 0.14),
                "w_soft": trial.suggest_float("w_soft", 0.00, 0.20),
                "w_zhong": trial.suggest_float("w_zhong", 0.02, 0.28),
                "w_ppr": trial.suggest_float("w_ppr", 0.00, 0.22),
                "w_fv": trial.suggest_float("w_fv", 0.00, 0.14),
                "w_vl": trial.suggest_float("w_vl", 0.00, 0.14),
                "w_sift": trial.suggest_float("w_sift", 0.004, 0.040),
            }
            return float(np.mean([_eval_stage1_mats(u, m, yk, kn, wd)[0] for (u, m, yk, kn) in fold_mats]))

        st_w = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=142))
        st_w.enqueue_trial(base_w)
        st_w.optimize(obj_weights, n_trials=n_weight_trials, show_progress_bar=False)
        best_c.update(st_w.best_params)
        opt_evals = [_eval_stage1_mats(u, m, yk, kn, st_w.best_params) for (u, m, yk, kn) in fold_mats]
        print(
            f"  [{sp}] 5-Fold Optuna A2 Stage-1: Top-12 Rec={np.mean([e[1] for e in opt_evals]):.5f}, "
            f"Known MAP@5={np.mean([e[2] for e in opt_evals]):.5f} (in {time.time() - tw:.1f}s)"
        )
        print("  Best Stage-1 cfg:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best_c.items()})
        best_cfgs[sp] = best_c

    with open(os.path.join(CACHE_V2_DIR, "optuna_stage1_best_cfg.json"), "w") as f:
        json.dump(best_cfgs, f, indent=2)
    return best_cfgs


def extract_and_cache_all_v2_bundles(best_cfgs, folds, train_df, test_df, all_identities, all_viewpoints, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr):
    """
    Precomputes and caches:
      1) Outer 5-Fold CV bundles (each outer fold's `val_bundle` + 4 nested inner `sub_bundles`!)
      2) Full-Train Production bundles (2 independent 5-subfold seeds + `test_bundle`!)
    Once cached, Optuna Stage B & Stage C tune 100+ trials across all 8 Stage-2 model families in minutes!
    """
    t0 = time.time()
    n_tr, n_te = len(train_df), len(test_df)
    test_glob = np.arange(n_tr, n_tr + n_te, dtype=np.int32)

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        cfg = best_cfgs[sp]
        outer_data = []
        rec12_list = []

        for fd in folds:
            fk = fd["fold"]
            gi, vi = fd["gallery_idx"], fd["val_idx"]
            g_rows = gi[train_df["species"].values[gi] == sp]
            vm = (train_df["species"].values[vi] == sp)
            v_rows = vi[vm]
            v_pos = np.where(vm)[0]
            y_v = fd["val_labels"][vm]
            is_n = fd["val_is_new"][vm]

            val_b = build_v2_feature_bundle(
                g_rows, v_rows, sp,
                all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                sift_csr, top_k_cands=12, stage1_cfg=cfg
            )
            val_b["y_true"] = y_v
            val_b["is_new"] = is_n
            val_b["v_pos"] = v_pos
            kn = ~is_n
            r12 = float(np.mean([y_v[kn][i] in val_b["cand_names"][kn][i] for i in range(int(kn.sum()))]))
            rec12_list.append(r12)

            # 4 internal encounter-grouped singleton-preserving sub-folds strictly inside `g_rows`!
            subfolds = build_open_set_splits(
                all_identities[g_rows], np.full(len(g_rows), sp, dtype=object), X_norm_all[g_rows],
                n_splits=4, enc_thresh=0.80, seed=101 + 37 * fk
            )
            sub_bundles = []
            for sf in subfolds:
                sg = g_rows[sf["gallery_idx"]]
                sq = g_rows[sf["val_idx"]]
                sb = build_v2_feature_bundle(
                    sg, sq, sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                    sift_csr, top_k_cands=12, stage1_cfg=cfg
                )
                sb["y_true"] = sf["val_labels"]
                sb["is_new"] = sf["val_is_new"]
                sub_bundles.append(sb)

            outer_data.append({"fold": fk, "val_bundle": val_b, "sub_bundles": sub_bundles})
            print(f"[{sp}] Extracted Outer Fold {fk} (val + 4 subfolds) | Known Top-12 Recall = {r12:.4f}")

        # Full-train production bundles (2 seeds x 5 subfolds + test_bundle)
        full_g = np.where(train_df["species"].values == sp)[0].astype(np.int32)
        tm = (test_df["species"].values == sp)
        te_pos = np.where(tm)[0]
        te_q = test_glob[tm]
        test_b = build_v2_feature_bundle(
            full_g, te_q, sp,
            all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
            sift_csr, top_k_cands=12, stage1_cfg=cfg
        )
        test_b["te_pos"] = te_pos

        full_seed_bundles = {}
        for sd in [2026, 7777]:
            sf_full = build_open_set_splits(
                all_identities[full_g], np.full(len(full_g), sp, dtype=object), X_norm_all[full_g],
                n_splits=5, enc_thresh=0.80, seed=sd
            )
            sbl = []
            for sf in sf_full:
                sg = full_g[sf["gallery_idx"]]
                sq = full_g[sf["val_idx"]]
                sb = build_v2_feature_bundle(
                    sg, sq, sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                    sift_csr, top_k_cands=12, stage1_cfg=cfg
                )
                sb["y_true"] = sf["val_labels"]
                sb["is_new"] = sf["val_is_new"]
                sbl.append(sb)
            full_seed_bundles[sd] = sbl

        out_path = os.path.join(CACHE_V2_DIR, f"v2_bundles_{sp}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({"outer_data": outer_data, "test_bundle": test_b, "full_seed_bundles": full_seed_bundles}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{sp.upper()}] Saved {out_path} in {time.time() - ts:.1f}s | Mean Outer Top-12 Known Recall = {np.mean(rec12_list):.4f}")

    print(f"\nAll V2 bundles extracted & cached in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    raw = load_all_raw_data()
    train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = raw
    folds = build_or_load_cv_v2()
    best_cfgs = tune_stage1_optuna(
        folds, train_df, all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts, sift_csr,
        n_geom_trials=14, n_weight_trials=90
    )
    extract_and_cache_all_v2_bundles(
        best_cfgs, folds, train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr
    )
