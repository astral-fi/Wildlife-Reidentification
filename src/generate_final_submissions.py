import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax, expit
from sklearn.preprocessing import normalize

from cv_framework import EST_PI_NEW, DATA_DIR
from pipeline_features import build_feature_bundle, CAND_FEATURE_NAMES
import exp04_grandmaster_ensemble as gm

warnings.filterwarnings("ignore")

# Override thread counts to 4 to avoid Windows OpenMP/MKL/CatBoost threadpool spinlock!
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def safe_cand_models(seed=42):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.4, max_iter=350, random_state=seed)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, alpha=2.5, max_iter=85, early_stopping=True, random_state=seed)),
        "lgb": lgb.LGBMClassifier(
            n_estimators=240, learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0, n_jobs=4, random_state=seed, verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=240, learning_rate=0.03, max_depth=4, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
            n_jobs=4, random_state=seed, eval_metric="logloss", verbosity=0
        ),
        "cat": CatBoostClassifier(
            iterations=280, learning_rate=0.035, depth=5, l2_leaf_reg=7.0,
            thread_count=4, random_seed=seed, verbose=0
        ),
    }


def safe_open_models(seed=42):
    return {
        "lgb": lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.03, num_leaves=12, max_depth=4,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=6.0, n_jobs=4, random_state=seed, verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=220, learning_rate=0.03, max_depth=3, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
            n_jobs=4, random_state=seed, eval_metric="logloss", verbosity=0
        ),
        "cat": CatBoostClassifier(
            iterations=260, learning_rate=0.035, depth=4, l2_leaf_reg=8.0,
            thread_count=4, random_seed=seed, verbose=0
        ),
    }


gm.make_cand_models = safe_cand_models
gm.make_open_models = safe_open_models


def borda_prob_blend_top5(pred_str_a, pred_str_b, w_a=0.55, w_b=0.45):
    """Reciprocal-rank MAP@5-optimal score blend of two ranked top-5 lists."""
    weights_rank = [1.0, 0.46, 0.28, 0.19, 0.14]
    sc = {}
    for rank_i, tok in enumerate(pred_str_a.split(" ")[:5]):
        sc[tok] = sc.get(tok, 0.0) + w_a * weights_rank[rank_i]
    for rank_i, tok in enumerate(pred_str_b.split(" ")[:5]):
        sc[tok] = sc.get(tok, 0.0) + w_b * weights_rank[rank_i]
    top5 = [k for k, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]]
    return " ".join(top5)


def main():
    t0 = time.time()
    os.makedirs("submissions", exist_ok=True)
    ART_DIR = r"C:\Users\ankit\.gemini\antigravity\brain\1b35347c-d2d2-4f03-9c8a-66a4dcb52159"

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
    offsets = np.load("cache/sift_offsets.npy")
    kp_counts = np.diff(offsets).astype(np.int32)

    sift_z = np.load("cache/sift_top36_global.npz")
    sift_nbr_idx = sift_z["nbr_idx"]
    sift_nbr_feats = sift_z["nbr_feats"]
    test_global_rows = np.arange(n_tr, n_tr + n_te, dtype=np.int32)

    exp04_pp_preds = [None] * n_te
    exp04_ind_preds = [None] * n_te
    feat_imps_by_sp = {}

    # Moderate empirically-validated gallery-size shift for Leopard (+0.28, half of outer shift +0.67)
    extra_open_shift = {"hyena": 0.0, "leopard": 0.28}

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        full_g_rows = np.where(train_df["species"].values == sp)[0].astype(np.int32)
        te_sp_mask = (test_df["species"].values == sp)
        te_pos = np.where(te_sp_mask)[0]
        te_q_rows = test_global_rows[te_sp_mask]

        test_bundle = build_feature_bundle(
            full_g_rows, te_q_rows, sp,
            all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
            sift_nbr_idx, sift_nbr_feats, top_k_cands=12
        )

        decs = []
        for idx_s, sd in enumerate([2026, 7777]):
            t_sd = time.time()
            if idx_s == 0:
                d_out, c_imp, _ = gm.train_species_grandmaster(
                    full_g_rows, [test_bundle], sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                    sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=5, seed=sd, return_importances=True
                )
                feat_imps_by_sp[sp] = c_imp
            else:
                d_out = gm.train_species_grandmaster(
                    full_g_rows, [test_bundle], sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                    sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=5, seed=sd, return_importances=False
                )
            decs.append(d_out[0])
            print(f"[{sp}] Finished seed {sd} in {time.time() - t_sd:.1f}s")

        top_imp = np.argsort(-feat_imps_by_sp[sp])[:12]
        print(f"[{sp.upper()}] Top-12 Features:", [(CAND_FEATURE_NAMES[j], round(float(feat_imps_by_sp[sp][j]), 1)) for j in top_imp])

        sh = extra_open_shift[sp]
        for key_cfg, out_arr in [("G2_Grandmaster_ClusterPP", exp04_pp_preds), ("G1_5Cand_GBDT3Open", exp04_ind_preds)]:
            p_new_ens = np.mean([expit(d[key_cfg]["raw_open_logits"] + d[key_cfg]["delta"] + sh) for d in decs], axis=0)
            cps = [softmax(d[key_cfg]["gamma"] * d[key_cfg]["raw_cand_logits"], axis=1) for d in decs]
            for loc_i, pos_te in enumerate(te_pos):
                p_n = float(p_new_ens[loc_i])
                sc_map = {"new_individual": p_n}
                w_seed = 1.0 / len(decs)
                for d_i, d in enumerate(decs):
                    c_names_d = d[key_cfg]["cand_names"]
                    for r in range(c_names_d.shape[1]):
                        cid = c_names_d[loc_i, r]
                        sc_map[cid] = sc_map.get(cid, 0.0) + w_seed * (1.0 - p_n) * float(cps[d_i][loc_i, r])
                top5 = [lbl for lbl, _ in sorted(sc_map.items(), key=lambda x: -x[1])[:5]]
                out_arr[pos_te] = " ".join(top5)
        print(f"Completed full [{sp}] test inference in {time.time() - ts:.1f}s")

    # Save Exp04 standalone submission AND Super-Blend with Exp03 (which achieved 0.91242 on Hyena!)
    sub_exp04 = pd.DataFrame({"id": test_df["id"].values, "predictions": exp04_pp_preds})
    sub_exp04.to_csv("submissions/submission_exp04_loio_grandmaster.csv", index=False)

    sub_exp03 = pd.read_csv("submissions/submission_best_full_ensemble_pp.csv")
    super_preds = []
    for i in range(n_te):
        sp = test_df["species"].iat[i]
        # Exp03 excelled on Hyena (0.91242 vs 0.91010); Exp04 excelled on Leopard (0.93230 vs 0.93106)!
        if sp == "hyena":
            super_preds.append(borda_prob_blend_top5(sub_exp03["predictions"].iat[i], exp04_pp_preds[i], w_a=0.60, w_b=0.40))
        else:
            super_preds.append(borda_prob_blend_top5(exp04_pp_preds[i], sub_exp03["predictions"].iat[i], w_a=0.65, w_b=0.35))

    sub_super_path = "submissions/submission_best.csv"
    pd.DataFrame({"id": test_df["id"].values, "predictions": super_preds}).to_csv(sub_super_path, index=False)
    # Also copy to root workspace as submission.csv for convenience!
    pd.DataFrame({"id": test_df["id"].values, "predictions": super_preds}).to_csv("submission.csv", index=False)

    gm.verify_submission_file(sub_super_path, test_df, train_df)
    gm.verify_submission_file("submissions/submission_exp04_loio_grandmaster.csv", test_df, train_df)
    gm.verify_submission_file("submissions/submission_best_full_ensemble_pp.csv", test_df, train_df)

    # Agreement & Rank-1 diagnostics across submissions
    s_best = pd.read_csv(sub_super_path)
    r1_best = s_best["predictions"].str.split(" ").str[0]
    r1_03 = sub_exp03["predictions"].str.split(" ").str[0]
    r1_04 = sub_exp04["predictions"].str.split(" ").str[0]
    print(f"Top-1 Agreement (Exp03 vs Exp04): {(r1_03 == r1_04).mean():.4f}")
    for sp in ["hyena", "leopard"]:
        m = (test_df["species"].values == sp)
        print(f"[SUPER-BLEND submission_best.csv | {sp}] Rank-1 `new_individual`={float((r1_best[m]=='new_individual').mean()):.3f} (est prior={EST_PI_NEW[sp]:.3f})")

    # Generate Model Progression & Feature Importance Plot in Artifact Directory!
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=150)

    stages = [
        "01 Raw Cosine\n+ Hard Thresh",
        "02 + Bayes-Opt\nMAP@5 Ranking",
        "03 + PCA-768\nPower Whitening",
        "04 + RootSIFT\n4-DOF RANSAC",
        "05 Stage-1 Multi-\nSpace + SIFT",
        "06 + LogReg\nStage-2 Stack",
        "07 + MLP Neural\n(From Scratch)",
        "08 + GBDT Trio\n(LGB/Cat/XGB)",
        "09 Grandmaster\nSuper-Ensemble",
    ]
    scores = [0.84369, 0.87401, 0.89281, 0.89487, 0.91254, 0.91583, 0.91815, 0.92269, 0.92363]
    colors = ["#8da0cb" if s < 0.885 else ("#66c2a5" if s < 0.92 else "#fc8d62") for s in scores]
    bars = axes[0].barh(np.arange(len(stages)), scores, color=colors, edgecolor="#333333", height=0.68)
    axes[0].axvline(0.88331, color="#e41a1c", ls="--", lw=1.8, label="Kaggle #1 Public LB (0.88331)")
    axes[0].set_yticks(np.arange(len(stages)))
    axes[0].set_yticklabels(stages, fontsize=8.8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.83, 0.942)
    axes[0].set_xlabel("5-Fold Open-Set CV LB-Proxy MAP@5")
    axes[0].set_title("A. End-to-End CV MAP@5 Progression (+8.00 pts)")
    for b, sc in zip(bars, scores):
        axes[0].text(sc + 0.0012, b.get_y() + b.get_height() / 2, f"{sc:.5f}", va="center", fontsize=8.5, fontweight="bold")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=8.5)

    # Panel B: Average Feature Importance across Hyena & Leopard
    avg_imp = 0.5 * (feat_imps_by_sp["hyena"] / feat_imps_by_sp["hyena"].sum()) + 0.5 * (feat_imps_by_sp["leopard"] / feat_imps_by_sp["leopard"].sum())
    top14 = np.argsort(-avg_imp)[:14]
    axes[1].barh(np.arange(14), avg_imp[top14] * 100, color="#377eb8", edgecolor="#222222", height=0.68)
    axes[1].set_yticks(np.arange(14))
    axes[1].set_yticklabels([CAND_FEATURE_NAMES[j] for j in top14], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Normalized Split Importance (%)")
    axes[1].set_title("B. Top-14 Stage-2 Reranker Features (Hyena + Leopard)")
    axes[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ART_DIR, "model_progression_and_importances.png"), dpi=150)
    plt.close()
    print(f"Saved plots & verified final submissions in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
