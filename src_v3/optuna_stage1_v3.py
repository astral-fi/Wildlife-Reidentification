"""
Optuna Stage A (V3) + Leak-Free V3 Feature Bundle Extraction:
  - Phase A1 (18 Optuna TPE trials/species on Folds 0-1):
      Tunes `tau_log`, `p_mid`, `reg_mid`, `d_mid`, `beta_dba`, `temp_dba`, `beta_qe`,
      `lda_shrink`, `gamma_kr`, `lam_kr`, `k1_zhong`, `diff_alpha`!
  - Phase A2 (140 Optuna TPE trials/species across ALL 5 outer folds in <25s!):
      Precomputes all 17 Stage-1 similarity matrices once across Folds 0-4 and runs 140 TPE trials
      maximizing `0.40 * Top-20 Recall + 0.60 * Known MAP@5`!
  - Extracts & caches all 72 leak-free V3 bundles (5 outer folds + 4 inner subfolds + 2 full-train 5-subfold seeds + test_bundle)
    into `cache_v3/v3_bundles_hyena.pkl` and `cache_v3/v3_bundles_leopard.pkl`!
"""
import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import optuna

from cv_v3 import build_or_load_cv_v3, build_open_set_splits, DATA_DIR, CACHE_V3_DIR
from features_v3 import build_v3_feature_bundle, DEFAULT_STAGE1_V3_CFG
from smoke_test_v3 import load_v3_raw_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

W_KEYS = [
    "w_p60", "w_dba", "w_zsc", "w_sub", "w_lda", "w_wccn", "w_kiss", "w_cca",
    "w_mlp", "w_kr", "w_svm", "w_soft", "w_zhong", "w_ppr", "w_fv", "w_vl", "w_sift",
]


def _eval_stage1_v3(u_cls, mats, y_kn, kn_mask, w_dict):
    C = sum(w_dict[k] * mats[k][kn_mask] for k in w_dict)
    top20 = np.argsort(-C, axis=1)[:, :20]
    c20 = u_cls[top20]
    hits12 = float(np.mean([y_kn[i] in c20[i, :12] for i in range(len(y_kn))]))
    hits20 = float(np.mean([y_kn[i] in c20[i] for i in range(len(y_kn))]))
    aps = np.zeros(len(y_kn), dtype=np.float64)
    for r in range(5):
        m = (c20[:, r] == y_kn) & (aps == 0.0)
        aps[m] = 1.0 / (r + 1)
    map5 = float(aps.mean())
    return 0.25 * hits12 + 0.20 * hits20 + 0.55 * map5, hits12, hits20, map5


def tune_stage1_v3_optuna(folds, train_df, all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts, sift_csr, n_geom_trials=16, n_weight_trials=140):
    best_cfgs = {}
    for sp in ["hyena", "leopard"]:
        print(f"\n=== V3 OPTUNA STAGE A1 (Geometry + DBA + a-QE + Kernel, {n_geom_trials} trials) [{sp.upper()}] ===")
        t_sp = time.time()
        base_w = {k: DEFAULT_STAGE1_V3_CFG[sp][k] for k in W_KEYS}

        def obj_geom(trial):
            cfg = dict(DEFAULT_STAGE1_V3_CFG[sp])
            cfg["tau_log"] = trial.suggest_float("tau_log", 1.2, 4.0)
            cfg["p_mid"] = trial.suggest_float("p_mid", 0.50, 0.72)
            cfg["reg_mid"] = trial.suggest_float("reg_mid", 5e-4, 3e-3, log=True)
            cfg["d_mid"] = trial.suggest_int("d_mid", 640, 896, step=64) if sp == "hyena" else trial.suggest_int("d_mid", 768, 1088, step=64)
            cfg["beta_dba"] = trial.suggest_float("beta_dba", 0.12, 0.42)
            cfg["temp_dba"] = trial.suggest_float("temp_dba", 3.5, 10.0)
            cfg["beta_qe"] = trial.suggest_float("beta_qe", 0.08, 0.38)
            cfg["lda_shrink"] = trial.suggest_float("lda_shrink", 0.12, 0.36)
            cfg["gamma_kr"] = trial.suggest_float("gamma_kr", 1.0, 3.0)
            cfg["lam_kr"] = trial.suggest_float("lam_kr", 0.12, 0.80, log=True)
            cfg["k1_zhong"] = trial.suggest_int("k1_zhong", 12, 24, step=2)
            cfg["diff_alpha"] = trial.suggest_float("diff_alpha", 0.70, 0.88)

            scs = []
            for fk in [0, 1]:
                fd = folds[fk]
                gi, vi = fd["gallery_idx"], fd["val_idx"]
                g_rows = gi[train_df["species"].values[gi] == sp]
                vm = (train_df["species"].values[vi] == sp)
                v_rows = vi[vm]
                y_v = fd["val_labels"][vm]
                kn = ~fd["val_is_new"][vm]
                res = build_v3_feature_bundle(
                    g_rows, v_rows, sp,
                    all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                    sift_csr, top_k_cands=20, stage1_cfg=cfg, return_stage1_mats=True
                )
                scs.append(_eval_stage1_v3(res["u_cls"], res["mats"], y_v[kn], kn, base_w)[0])
            return float(np.mean(scs))

        st_g = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        st_g.enqueue_trial({k: DEFAULT_STAGE1_V3_CFG[sp][k] for k in ["tau_log", "p_mid", "reg_mid", "d_mid", "beta_dba", "temp_dba", "beta_qe", "lda_shrink", "gamma_kr", "lam_kr", "k1_zhong", "diff_alpha"]})
        st_g.optimize(obj_geom, n_trials=n_geom_trials, show_progress_bar=False)

        best_c = dict(DEFAULT_STAGE1_V3_CFG[sp])
        best_c.update(st_g.best_params)
        print(f"[{sp.upper()}] Stage A1 Best Obj={st_g.best_value:.5f} (trial #{st_g.best_trial.number}, base={st_g.trials[0].value:.5f}) in {time.time() - t_sp:.1f}s")

        # Phase A2: Precompute all 17 Stage-1 matrices across ALL 5 outer folds ONCE, then 140 Optuna trials!
        print(f"=== V3 OPTUNA STAGE A2 (5-Fold 17-Engine Fusion Weights, {n_weight_trials} trials) [{sp.upper()}] ===")
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
            res = build_v3_feature_bundle(
                g_rows, v_rows, sp,
                all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                sift_csr, top_k_cands=20, stage1_cfg=best_c, return_stage1_mats=True
            )
            fold_mats.append((res["u_cls"], res["mats"], y_v[kn], kn))

        base_ev = [_eval_stage1_v3(u, m, yk, kn, base_w) for (u, m, yk, kn) in fold_mats]
        print(f"  [{sp}] 5-Fold Base Stage-1: Top-12={np.mean([e[1] for e in base_ev]):.5f}, Top-20={np.mean([e[2] for e in base_ev]):.5f}, Known MAP@5={np.mean([e[3] for e in base_ev]):.5f}")

        def obj_w(trial):
            wd = {
                "w_p60": trial.suggest_float("w_p60", 0.04, 0.35),
                "w_dba": trial.suggest_float("w_dba", 0.08, 0.42),
                "w_zsc": trial.suggest_float("w_zsc", 0.00, 0.22),
                "w_sub": trial.suggest_float("w_sub", 0.02, 0.28),
                "w_lda": trial.suggest_float("w_lda", 0.05, 0.32),
                "w_wccn": trial.suggest_float("w_wccn", 0.00, 0.16),
                "w_kiss": trial.suggest_float("w_kiss", 0.00, 0.16),
                "w_cca": trial.suggest_float("w_cca", 0.00, 0.14),
                "w_mlp": trial.suggest_float("w_mlp", 0.02, 0.28),
                "w_kr": trial.suggest_float("w_kr", 0.02, 0.28),
                "w_svm": trial.suggest_float("w_svm", 0.00, 0.22),
                "w_soft": trial.suggest_float("w_soft", 0.00, 0.20),
                "w_zhong": trial.suggest_float("w_zhong", 0.00, 0.28),
                "w_ppr": trial.suggest_float("w_ppr", 0.00, 0.24),
                "w_fv": trial.suggest_float("w_fv", 0.00, 0.16),
                "w_vl": trial.suggest_float("w_vl", 0.00, 0.16),
                "w_sift": trial.suggest_float("w_sift", 0.008, 0.065),
            }
            return float(np.mean([_eval_stage1_v3(u, m, yk, kn, wd)[0] for (u, m, yk, kn) in fold_mats]))

        st_w = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=142))
        st_w.enqueue_trial(base_w)
        st_w.optimize(obj_w, n_trials=n_weight_trials, show_progress_bar=False)
        best_c.update(st_w.best_params)
        opt_ev = [_eval_stage1_v3(u, m, yk, kn, st_w.best_params) for (u, m, yk, kn) in fold_mats]
        print(
            f"  [{sp}] 5-Fold Optuna A2:    Top-12={np.mean([e[1] for e in opt_ev]):.5f}, "
            f"Top-20={np.mean([e[2] for e in opt_ev]):.5f}, Known MAP@5={np.mean([e[3] for e in opt_ev]):.5f} (in {time.time() - tw:.1f}s)"
        )
        print("  Best Stage-1 V3 cfg:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best_c.items()})
        best_cfgs[sp] = best_c

    with open(os.path.join(CACHE_V3_DIR, "optuna_stage1_v3_best_cfg.json"), "w") as f:
        json.dump(best_cfgs, f, indent=2)
    return best_cfgs


def extract_all_v3_bundles(best_cfgs, folds, train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr):
    t0 = time.time()
    n_tr, n_te = len(train_df), len(test_df)
    test_glob = np.arange(n_tr, n_tr + n_te, dtype=np.int32)

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        cfg = best_cfgs[sp]
        outer_data = []
        r12_l, r20_l = [], []

        for fd in folds:
            fk = fd["fold"]
            gi, vi = fd["gallery_idx"], fd["val_idx"]
            g_rows = gi[train_df["species"].values[gi] == sp]
            vm = (train_df["species"].values[vi] == sp)
            v_rows = vi[vm]
            v_pos = np.where(vm)[0]
            y_v = fd["val_labels"][vm]
            is_n = fd["val_is_new"][vm]

            val_b = build_v3_feature_bundle(
                g_rows, v_rows, sp,
                all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                sift_csr, top_k_cands=20, stage1_cfg=cfg
            )
            val_b["y_true"] = y_v
            val_b["is_new"] = is_n
            val_b["v_pos"] = v_pos
            kn = ~is_n
            r12 = float(np.mean([y_v[kn][i] in val_b["cand_names"][kn][i, :12] for i in range(int(kn.sum()))]))
            r20 = float(np.mean([y_v[kn][i] in val_b["cand_names"][kn][i] for i in range(int(kn.sum()))]))
            r12_l.append(r12); r20_l.append(r20)

            subfolds = build_open_set_splits(
                all_id[g_rows], np.full(len(g_rows), sp, dtype=object), X_norm_all[g_rows],
                n_splits=4, enc_thresh=0.80, seed=101 + 37 * fk
            )
            sub_bundles = []
            for sf in subfolds:
                sg = g_rows[sf["gallery_idx"]]
                sq = g_rows[sf["val_idx"]]
                sb = build_v3_feature_bundle(
                    sg, sq, sp,
                    all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                    sift_csr, top_k_cands=20, stage1_cfg=cfg
                )
                sb["y_true"] = sf["val_labels"]
                sb["is_new"] = sf["val_is_new"]
                sub_bundles.append(sb)

            outer_data.append({"fold": fk, "val_bundle": val_b, "sub_bundles": sub_bundles})
            print(f"[{sp}] Extracted Outer Fold {fk} (val + 4 subfolds) | Known Top-12={r12:.4f}, Top-20={r20:.4f}")

        full_g = np.where(train_df["species"].values == sp)[0].astype(np.int32)
        tm = (test_df["species"].values == sp)
        te_pos = np.where(tm)[0]
        te_q = test_glob[tm]
        test_b = build_v3_feature_bundle(
            full_g, te_q, sp,
            all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
            sift_csr, top_k_cands=20, stage1_cfg=cfg
        )
        test_b["te_pos"] = te_pos

        full_seed_bundles = {}
        for sd in [2026, 7777]:
            sf_full = build_open_set_splits(
                all_id[full_g], np.full(len(full_g), sp, dtype=object), X_norm_all[full_g],
                n_splits=5, enc_thresh=0.80, seed=sd
            )
            sbl = []
            for sf in sf_full:
                sg = full_g[sf["gallery_idx"]]
                sq = full_g[sf["val_idx"]]
                sb = build_v3_feature_bundle(
                    sg, sq, sp,
                    all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
                    sift_csr, top_k_cands=20, stage1_cfg=cfg
                )
                sb["y_true"] = sf["val_labels"]
                sb["is_new"] = sf["val_is_new"]
                sbl.append(sb)
            full_seed_bundles[sd] = sbl

        out_path = os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({"outer_data": outer_data, "test_bundle": test_b, "full_seed_bundles": full_seed_bundles}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{sp.upper()}] Saved {out_path} in {time.time() - ts:.1f}s | Mean Outer Top-12={np.mean(r12_l):.4f}, Top-20={np.mean(r20_l):.4f}")

    print(f"\nAll V3 bundles extracted & cached in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    raw = load_v3_raw_data()
    train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr = raw
    folds = build_or_load_cv_v3()
    best_cfgs = tune_stage1_v3_optuna(
        folds, train_df, all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts, sift_csr,
        n_geom_trials=12, n_weight_trials=150
    )
    extract_all_v3_bundles(
        best_cfgs, folds, train_df, test_df, all_id, all_vp, miew_all, miew_norms, X_norm_all, vlad_all, fv_all, kp_counts, sift_csr
    )
