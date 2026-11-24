"""
Shift-Robust V2 Champion Pipeline (Zero Feature-Distribution Shift Between CV & Test!):
  - Keeps ALL 9 Peak V2 Candidate Rankers (LGBMRanker, XGBRanker, CatBoostRanker(QuerySoftMax),
    LGBM/XGB/Cat/MLP/LogReg binary, Stage-1) using the Optuna-tuned hyperparameters.
  - Fixes the Leopard CV->Test domain shift discovered by `diagnose_v2.py`:
      1) Drops gallery-size-dependent raw background columns (`*_bg`, `*_s2`, `hub0`, `log_freq0`)
         and burst-size-dependent query-density columns (`diff_q_enrich`, `top5/10_q_enrich`,
         `clust_mean_q_enrich`, `log_clust_size`, `qq_gap_*`, `max_qq_*`) from the Open-Set detector!
      2) Keeps ALL domain-invariant LOIO-calibrated scores (`z45/z60/zraw/zlda/zkiss`, `dq*`, `ec*`),
         relative gaps (`g12/g14/g16`), cluster-vs-gallery scores (`clust_max_z60`, `clust_mean_z60`,
         `z60_minus_cmax`, `clust_max_sim_inl`), SIFT + Viewpoint, and adds background-trap-proof
         `sift_x_z60 = g_sim0 * max(0, z60)` and `fund_x_z60 = g_fnd0 * max(0, z60)`!
      3) Re-tunes Open-Set LGBM, XGB, CatBoost via Optuna TPE on the shift-free features.
      4) Runs strict Nested Leave-One-Outer-Fold-Out Optuna Stage C (with coherent `gamma >= 1.8`
         and 1-hop + mild CC consolidation) and 260-model multi-seed full-train inference!
"""
import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import optuna
from scipy.special import softmax, expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from cv_v2 import build_or_load_cv_v2, eval_cv_v2, PRIOR_PI_NEW, DATA_DIR, CACHE_V2_DIR
from optuna_stage2_and_train import (
    clip_logit, row_zscore, verify_submission_file, build_cand_dataset,
    precompute_graph_struct, apply_two_level_graph_postproc, saerens_em_prior,
)
from diagnose_v2 import OPEN_NAMES

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columns that shift heavily with gallery size (n_g) or mega-identity burst size (n_q) between CV and test.csv!
SHIFT_PRONE_OPEN_COLS = {
    "st1_s2", "st1_bg",
    "p60_s2", "p60_bg",
    "p45_s2",
    "raw_s2",
    "sub_s2",
    "lda_s2",
    "kiss_s1", "kiss_s2",
    "zh_s2",
    "ppr_s2",
    "fv_s2",
    "vl_s2",
    "log_freq0", "hub0",
    "diff_q_enrich", "top5_q_enrich", "top10_q_enrich", "clust_mean_q_enrich",
    "max_qq_mix", "max_qq_raw", "qq_gap_mix", "qq_gap_raw", "max_qq_sift",
    "log_clust_size",
}

KEEP_OPEN_IDX = [i for i, n in enumerate(OPEN_NAMES) if n not in SHIFT_PRONE_OPEN_COLS]
IDX_Z60 = OPEN_NAMES.index("z60")
IDX_GSIM0 = OPEN_NAMES.index("g_sim0")
IDX_GFND0 = OPEN_NAMES.index("g_fnd0")
IDX_SIFT_MAX = OPEN_NAMES.index("sift_sim_max")


def make_shift_robust_open_base(X_open_raw):
    """Extracts 73 domain-invariant open-set features + 3 SIFT-x-MiewID background-trap-proof interaction features!"""
    base = X_open_raw[:, KEEP_OPEN_IDX]
    z60_pos = np.maximum(0.0, X_open_raw[:, IDX_Z60])
    sift_x_z60 = X_open_raw[:, IDX_GSIM0] * z60_pos
    fund_x_z60 = X_open_raw[:, IDX_GFND0] * z60_pos
    sift_max_x_z60 = X_open_raw[:, IDX_SIFT_MAX] * z60_pos
    return np.column_stack([base, sift_x_z60, fund_x_z60, sift_max_x_z60]).astype(np.float32)


def _apply_ultra_guard(p_new, raw_s1, max_sim_inl, z60):
    if raw_s1 is None or max_sim_inl is None:
        return p_new
    ultra = (raw_s1 >= 0.78) & (max_sim_inl >= 12.0)
    if z60 is not None:
        ultra = ultra | ((raw_s1 >= 0.56) & (z60 >= 2.0) & (max_sim_inl >= 25.0))
    return np.where(ultra, np.minimum(p_new, 0.10), p_new)


def open_map5_from_probs_v2(cand_names, cand_probs, open_logits, y_true, is_new, pi_new, delta=0.0, raw_s1=None, max_sim_inl=None, z60=None):
    w = np.where(is_new, pi_new / max(1, int(is_new.sum())), (1.0 - pi_new) / max(1, int((~is_new).sum())))
    ord5 = np.argsort(-cand_probs, axis=1)[:, :5]
    top5_names = np.take_along_axis(cand_names, ord5, axis=1)
    cp5 = np.take_along_axis(cand_probs, ord5, axis=1)

    p_new = _apply_ultra_guard(expit(open_logits + delta), raw_s1, max_sim_inl, z60)
    kp = (1.0 - p_new[:, None]) * cp5
    rank_new = 1 + np.sum(kp > p_new[:, None], axis=1)
    ap_new = np.where(rank_new <= 5, 1.0 / rank_new, 0.0)

    hit_col = np.full(len(y_true), -1, dtype=np.int32)
    for r in range(5):
        m = (top5_names[:, r] == y_true) & (hit_col < 0)
        hit_col[m] = r
    vh = np.where(hit_col >= 0)[0]
    hc = hit_col[vh]
    ap_kn = np.zeros(len(y_true), dtype=np.float64)
    rk_hit = (hc + 1) + (p_new[vh] >= kp[vh, hc]).astype(np.int32)
    ap_kn[vh] = np.where(rk_hit <= 5, 1.0 / rk_hit, 0.0)
    return float(np.sum(np.where(is_new, ap_new, ap_kn) * w))


def decode_top5_from_probs_v2(cand_names, cand_probs, open_logits, delta=0.0, raw_s1=None, max_sim_inl=None, z60=None):
    ord5 = np.argsort(-cand_probs, axis=1)[:, :5]
    top5_names = np.take_along_axis(cand_names, ord5, axis=1)
    cp5 = np.take_along_axis(cand_probs, ord5, axis=1)
    p_new = _apply_ultra_guard(expit(open_logits + delta), raw_s1, max_sim_inl, z60)

    preds = []
    for i in range(len(cand_names)):
        pn = float(p_new[i])
        kp = (1.0 - pn) * cp5[i]
        items = [("new_individual", pn)] + [(top5_names[i, r], float(kp[r])) for r in range(5)]
        items.sort(key=lambda x: -x[1])
        preds.append([x[0] for x in items[:5]])
    return preds


def optuna_tune_shift_robust_open(species, full_sub_bundles, n_trials=16):
    pi_new = PRIOR_PI_NEW[species]
    eval_subs = [0, 1, 2]
    X_sr_list = [make_shift_robust_open_base(sb["X_open_base"]) for sb in full_sub_bundles]

    def build_open_tr(hs):
        Xo, yo, wo = [], [], []
        for idx_s, sb in enumerate(full_sub_bundles):
            if idx_s == hs:
                continue
            isn = sb["is_new"]
            Xo.append(X_sr_list[idx_s])
            yo.append(isn.astype(np.int32))
            nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
            wo.append(np.where(isn, pi_new / nn, (1.0 - pi_new) / nk) * len(isn))
        return np.vstack(Xo), np.concatenate(yo), np.concatenate(wo)

    def eval_cal(hb, pr):
        isn = hb["is_new"]
        y_i = isn.astype(int)
        nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
        w_i = np.where(isn, pi_new / nn, (1.0 - pi_new) / nk)
        return float(roc_auc_score(y_i, pr) - 0.35 * log_loss(y_i, np.clip(pr, 1e-5, 1.0 - 1e-5), sample_weight=w_i))

    out = {}

    def obj_lgb(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 140, 340, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 24),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 55),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.5, 22.0, log=True),
        }
        scs = []
        for hs in eval_subs:
            Xo, yo, wo = build_open_tr(hs)
            m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=42 + hs, verbose=-1, **p)
            m.fit(Xo, yo, sample_weight=wo)
            scs.append(eval_cal(full_sub_bundles[hs], m.predict_proba(X_sr_list[hs])[:, 1]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=146))
    st.optimize(obj_lgb, n_trials=n_trials)
    out["lgb_open"] = st.best_params
    print(f"  [{species}] Shift-Robust Optuna LGBM Open-Set     Best Cal-Obj = {st.best_value:.5f}")

    def obj_xgb(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 140, 340, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 4.0, 22.0),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.5, 22.0, log=True),
        }
        scs = []
        for hs in eval_subs:
            Xo, yo, wo = build_open_tr(hs)
            m = xgb.XGBClassifier(n_jobs=4, random_state=42 + hs, eval_metric="logloss", verbosity=0, **p)
            m.fit(Xo, yo, sample_weight=wo)
            scs.append(eval_cal(full_sub_bundles[hs], m.predict_proba(X_sr_list[hs])[:, 1]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=147))
    st.optimize(obj_xgb, n_trials=n_trials)
    out["xgb_open"] = st.best_params
    print(f"  [{species}] Shift-Robust Optuna XGB Open-Set      Best Cal-Obj = {st.best_value:.5f}")

    def obj_cat(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 180, 360, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.07, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 20.0, log=True),
        }
        scs = []
        for hs in eval_subs[:2]:
            Xo, yo, wo = build_open_tr(hs)
            m = CatBoostClassifier(thread_count=4, random_seed=42 + hs, verbose=0, **p)
            m.fit(Xo, yo, sample_weight=wo)
            scs.append(eval_cal(full_sub_bundles[hs], m.predict_proba(X_sr_list[hs])[:, 1]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=148))
    st.optimize(obj_cat, n_trials=max(10, n_trials // 2))
    out["cat_open"] = st.best_params
    print(f"  [{species}] Shift-Robust Optuna CatBoost Open-Set Best Cal-Obj = {st.best_value:.5f}")
    return out


def fit_all_shift_robust_models(sub_bundles, target_bundle, species, hp, seed=42):
    pi_new = PRIOR_PI_NEW[species]
    n_sf = len(sub_bundles)
    ckeys = ["lgb_rank", "xgb_rank", "cat_rank", "lgb_bin", "xgb_bin", "cat_bin", "mlp_bin", "logreg_bin"]
    okeys = ["lgb_open", "xgb_open", "cat_open", "mlp_open", "logreg_open"]

    nt, Kt, Ft = target_bundle["X_cand"].shape
    X_t_c = target_bundle["X_cand"].reshape(-1, Ft)
    oof_c = {k: [] for k in ckeys}
    tgt_c = {k: np.zeros((nt, Kt), dtype=np.float64) for k in ckeys}

    for hs in range(n_sf):
        hb = sub_bundles[hs]
        nh, Kh, Fh = hb["X_cand"].shape
        X_h_c = hb["X_cand"].reshape(-1, Fh)

        Xtr_r, ytr_r, gtr_r = build_cand_dataset(sub_bundles, hs, for_ranker=True)
        qids_r = np.repeat(np.arange(len(gtr_r)), gtr_r)
        Xtr_b, ytr_b, _ = build_cand_dataset(sub_bundles, hs, for_ranker=False)

        m = lgb.LGBMRanker(objective="lambdarank", eval_at=[1, 5], subsample_freq=1, n_jobs=4, random_state=seed + 11 * hs, verbose=-1, **hp["lgb_rank"])
        m.fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["lgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["lgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = xgb.XGBRanker(objective="rank:ndcg", n_jobs=4, random_state=seed + 13 * hs, verbosity=0, **hp["xgb_rank"])
        m.fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["xgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["xgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = CatBoostRanker(loss_function="QuerySoftMax", thread_count=4, random_seed=seed + 17 * hs, verbose=0, **hp["cat_rank"])
        m.fit(Pool(data=Xtr_r, label=ytr_r.astype(np.float32), group_id=qids_r))
        oof_c["cat_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["cat_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 19 * hs, verbose=-1, **hp["lgb_bin"])
        m.fit(Xtr_b, ytr_b)
        oof_c["lgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["lgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 23 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_bin"])
        m.fit(Xtr_b, ytr_b)
        oof_c["xgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["xgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = CatBoostClassifier(
            iterations=hp["cat_rank"]["iterations"], learning_rate=hp["cat_rank"]["learning_rate"],
            depth=hp["cat_rank"]["depth"], l2_leaf_reg=hp["cat_rank"]["l2_leaf_reg"],
            thread_count=4, random_seed=seed + 29 * hs, verbose=0
        )
        m.fit(Xtr_b, ytr_b)
        oof_c["cat_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["cat_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, alpha=2.5, max_iter=85, early_stopping=True, random_state=seed + 31 * hs))
        m.fit(Xtr_b, ytr_b)
        oof_c["mlp_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["mlp_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=300, random_state=seed + 37 * hs))
        m.fit(Xtr_b, ytr_b)
        oof_c["logreg_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["logreg_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

    def make_stacked_open(X_base_raw, bin_mat, rank_mat):
        X_sr = make_shift_robust_open_base(X_base_raw)
        sb = -np.sort(-bin_mat, axis=1)
        sr = -np.sort(-rank_mat, axis=1)
        pb = softmax(bin_mat, axis=1)
        pr = softmax(1.5 * rank_mat, axis=1)
        eb = -np.sum(pb * np.log(np.clip(pb, 1e-12, None)), axis=1)
        er = -np.sum(pr * np.log(np.clip(pr, 1e-12, None)), axis=1)
        return np.column_stack([X_sr, sb[:, 0], sb[:, 0] - sb[:, 1], pb.max(1), eb, sr[:, 0], sr[:, 0] - sr[:, 1], pr.max(1), er]).astype(np.float32)

    sub_open_st = []
    for s in range(n_sf):
        b_avg = 0.28 * oof_c["lgb_bin"][s] + 0.28 * oof_c["xgb_bin"][s] + 0.24 * oof_c["cat_bin"][s] + 0.10 * oof_c["mlp_bin"][s] + 0.10 * oof_c["logreg_bin"][s]
        r_avg = (oof_c["lgb_rank"][s] + oof_c["xgb_rank"][s] + oof_c["cat_rank"][s]) / 3.0
        sub_open_st.append(make_stacked_open(sub_bundles[s]["X_open_base"], b_avg, r_avg))

    t_b_avg = 0.28 * tgt_c["lgb_bin"] + 0.28 * tgt_c["xgb_bin"] + 0.24 * tgt_c["cat_bin"] + 0.10 * tgt_c["mlp_bin"] + 0.10 * tgt_c["logreg_bin"]
    t_r_avg = (tgt_c["lgb_rank"] + tgt_c["xgb_rank"] + tgt_c["cat_rank"]) / 3.0
    X_t_open = make_stacked_open(target_bundle["X_open_base"], t_b_avg, t_r_avg)

    tgt_o = {k: np.zeros(nt, dtype=np.float64) for k in okeys}
    for hs in range(n_sf):
        Xo_l, yo_l, wo_l = [], [], []
        for idx_s, sb in enumerate(sub_bundles):
            if idx_s == hs:
                continue
            isn = sb["is_new"]
            Xo_l.append(sub_open_st[idx_s])
            yo_l.append(isn.astype(np.int32))
            nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
            wo_l.append(np.where(isn, pi_new / nn, (1.0 - pi_new) / nk) * len(isn))
        Xo_tr, yo_tr, wo_tr = np.vstack(Xo_l), np.concatenate(yo_l), np.concatenate(wo_l)

        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 41 * hs, verbose=-1, **hp["lgb_open"])
        m.fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["lgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 43 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_open"])
        m.fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["xgb_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

        m = CatBoostClassifier(thread_count=4, random_seed=seed + 47 * hs, verbose=0, **hp["cat_open"])
        m.fit(Xo_tr, yo_tr, sample_weight=wo_tr)
        tgt_o["cat_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(48, 24), batch_size=256, alpha=3.5, max_iter=100, early_stopping=True, random_state=seed + 53 * hs))
        m.fit(Xo_tr, yo_tr)
        ps = logit(pi_new) - logit(float(yo_tr.mean()))
        tgt_o["mlp_open"] += (clip_logit(m.predict_proba(X_t_open)[:, 1]) + ps) / n_sf

        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.4, max_iter=300, random_state=seed + 59 * hs))
        m.fit(Xo_tr, yo_tr, logisticregression__sample_weight=wo_tr)
        tgt_o["logreg_open"] += clip_logit(m.predict_proba(X_t_open)[:, 1]) / n_sf

    tgt_c["stage1"] = row_zscore(target_bundle["stage1_top_scores"])
    return tgt_c, tgt_o


def optuna_meta_shift_robust(outer_preds_list, species, n_trials=120):
    pi_new = PRIOR_PI_NEW[species]
    for item in outer_preds_list:
        b = item["bundle"]
        item["graph_struct"] = precompute_graph_struct(b["W_qq"], b["w_qq_sum"], b["has_qq"], b["q_comp"], b["nc_q"])
        item["z60"] = b["X_open_base"][:, IDX_Z60]

    def combine_fold(fd_item, p):
        c_bin = (
            p["w_lgb_bin"] * fd_item["cand"]["lgb_bin"]
            + p["w_xgb_bin"] * fd_item["cand"]["xgb_bin"]
            + p["w_cat_bin"] * fd_item["cand"]["cat_bin"]
            + p["w_mlp_bin"] * fd_item["cand"]["mlp_bin"]
            + p["w_logreg_bin"] * fd_item["cand"]["logreg_bin"]
        )
        c_rnk = (
            p["w_stage1"] * fd_item["cand"]["stage1"]
            + p["w_lgb_rank"] * fd_item["cand"]["lgb_rank"]
            + p["w_xgb_rank"] * fd_item["cand"]["xgb_rank"]
            + p["w_cat_rank"] * fd_item["cand"]["cat_rank"]
        )
        c_tot = (1.0 - p["mix_rank"]) * row_zscore(c_bin) + p["mix_rank"] * row_zscore(c_rnk)
        o_tot = (
            p["wo_lgb"] * fd_item["open"]["lgb_open"]
            + p["wo_xgb"] * fd_item["open"]["xgb_open"]
            + p["wo_cat"] * fd_item["open"]["cat_open"]
            + p["wo_mlp"] * fd_item["open"]["mlp_open"]
            + p["wo_lr"] * fd_item["open"]["logreg_open"]
        ) / max(1e-6, p["wo_lgb"] + p["wo_xgb"] + p["wo_cat"] + p["wo_mlp"] + p["wo_lr"])

        b = fd_item["bundle"]
        return apply_two_level_graph_postproc(
            b["cand_names"], c_tot, o_tot, b["W_qq"], b["w_qq_sum"], b["has_qq"], b["q_comp"], b["nc_q"],
            gamma=p["gamma"], alpha_open_1h=p["a_o1"], alpha_cand_1h=p["a_c1"],
            alpha_open_cc=p["a_oc"], alpha_cand_cc=p["a_cc"],
            graph_struct=fd_item.get("graph_struct")
        )

    def suggest_meta_params(trial):
        return {
            "w_stage1": trial.suggest_float("w_stage1", 0.0, 0.35),
            "w_lgb_rank": trial.suggest_float("w_lgb_rank", 0.05, 0.50),
            "w_xgb_rank": trial.suggest_float("w_xgb_rank", 0.05, 0.50),
            "w_cat_rank": trial.suggest_float("w_cat_rank", 0.05, 0.50),
            "w_lgb_bin": trial.suggest_float("w_lgb_bin", 0.05, 0.50),
            "w_xgb_bin": trial.suggest_float("w_xgb_bin", 0.05, 0.50),
            "w_cat_bin": trial.suggest_float("w_cat_bin", 0.05, 0.50),
            "w_mlp_bin": trial.suggest_float("w_mlp_bin", 0.0, 0.30),
            "w_logreg_bin": trial.suggest_float("w_logreg_bin", 0.0, 0.35),
            "mix_rank": trial.suggest_float("mix_rank", 0.15, 0.70),
            "wo_lgb": trial.suggest_float("wo_lgb", 0.10, 0.50),
            "wo_xgb": trial.suggest_float("wo_xgb", 0.10, 0.50),
            "wo_cat": trial.suggest_float("wo_cat", 0.10, 0.50),
            "wo_mlp": trial.suggest_float("wo_mlp", 0.0, 0.25),
            "wo_lr": trial.suggest_float("wo_lr", 0.0, 0.20),
            "a_o1": trial.suggest_float("a_o1", 0.05, 0.40),
            "a_c1": trial.suggest_float("a_c1", 0.05, 0.35),
            "a_oc": trial.suggest_float("a_oc", 0.0, 0.18),   # bounded CC pooling prevents giant-cluster drift!
            "a_cc": trial.suggest_float("a_cc", 0.0, 0.18),
            "gamma": trial.suggest_float("gamma", 2.0, 4.5),  # keeps new_individual cleanly in Rank 1-2!
            "delta": trial.suggest_float("delta", -0.60, 0.90),
        }

    base_meta = {
        "w_stage1": 0.12, "w_lgb_rank": 0.30, "w_xgb_rank": 0.30, "w_cat_rank": 0.30,
        "w_lgb_bin": 0.28, "w_xgb_bin": 0.28, "w_cat_bin": 0.24, "w_mlp_bin": 0.10, "w_logreg_bin": 0.10,
        "mix_rank": 0.45,
        "wo_lgb": 0.30, "wo_xgb": 0.30, "wo_cat": 0.25, "wo_mlp": 0.08, "wo_lr": 0.07,
        "a_o1": 0.25, "a_c1": 0.20, "a_oc": 0.06, "a_cc": 0.06,
        "gamma": 2.8, "delta": 0.20,
    }

    nested_preds_by_fold = {}
    for hold_f in range(5):
        train_fs = [outer_preds_list[f] for f in range(5) if f != hold_f]

        def obj_meta(trial):
            p = suggest_meta_params(trial)
            return float(np.mean([
                open_map5_from_probs_v2(*combine_fold(item, p), item["bundle"]["y_true"], item["bundle"]["is_new"], pi_new, delta=p["delta"], raw_s1=item["bundle"]["raw_s1"], max_sim_inl=item["bundle"]["max_sim_inl"], z60=item["z60"])
                for item in train_fs
            ]))

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=200 + hold_f))
        st.enqueue_trial(base_meta)
        st.optimize(obj_meta, n_trials=n_trials)
        bp = st.best_params
        val_item = outer_preds_list[hold_f]
        cn_v, cp_v, ol_v = combine_fold(val_item, bp)
        bv = val_item["bundle"]
        nested_preds_by_fold[hold_f] = decode_top5_from_probs_v2(cn_v, cp_v, ol_v, delta=bp["delta"], raw_s1=bv["raw_s1"], max_sim_inl=bv["max_sim_inl"], z60=val_item["z60"])

    def obj_full(trial):
        p = suggest_meta_params(trial)
        return float(np.mean([
            open_map5_from_probs_v2(*combine_fold(item, p), item["bundle"]["y_true"], item["bundle"]["is_new"], pi_new, delta=p["delta"], raw_s1=item["bundle"]["raw_s1"], max_sim_inl=item["bundle"]["max_sim_inl"], z60=item["z60"])
            for item in outer_preds_list
        ]))

    st_full = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2027))
    st_full.enqueue_trial(base_meta)
    st_full.optimize(obj_full, n_trials=n_trials + 40)
    full_meta_p = st_full.best_params
    print(f"[{species.upper()}] Shift-Robust Stage C (5-Fold All) MAP@5={st_full.best_value:.5f} | mix_rank={full_meta_p['mix_rank']:.2f}, gamma={full_meta_p['gamma']:.2f}, delta={full_meta_p['delta']:+.3f}, a_o1={full_meta_p['a_o1']:.2f}, a_oc={full_meta_p['a_oc']:.2f}")
    return nested_preds_by_fold, full_meta_p, combine_fold


def main():
    t0 = time.time()
    os.makedirs("submissions_v2", exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    folds = build_or_load_cv_v2()
    n_te = len(test_df)

    with open(os.path.join(CACHE_V2_DIR, "optuna_stage2_best_params.json"), "r") as f:
        prev_params = json.load(f)

    sr_hp_all = {}
    sr_meta_all = {}
    nested_cv_preds_sr = {k: [None] * len(folds[k]["val_idx"]) for k in range(5)}
    test_preds_sr = [None] * n_te

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        with open(os.path.join(CACHE_V2_DIR, f"v2_bundles_{sp}.pkl"), "rb") as f:
            data = pickle.load(f)

        outer_data = data["outer_data"]
        test_b = data["test_bundle"]
        full_seed_bundles = data["full_seed_bundles"]

        hp = dict(prev_params["stage2_hp"][sp])
        print(f"\n=== [{sp.upper()}] OPTUNA SHIFT-ROBUST OPEN-SET HPO (76 domain-invariant + 8 stacked features) ===")
        open_sr_hp = optuna_tune_shift_robust_open(sp, full_seed_bundles[2026], n_trials=16)
        hp.update(open_sr_hp)
        sr_hp_all[sp] = hp

        outer_preds_list = []
        for fd_data in outer_data:
            fk = fd_data["fold"]
            cand_d, open_d = fit_all_shift_robust_models(fd_data["sub_bundles"], fd_data["val_bundle"], sp, hp, seed=42 + 100 * fk)
            outer_preds_list.append({"fold": fk, "bundle": fd_data["val_bundle"], "cand": cand_d, "open": open_d})

        nested_sp, full_meta_p, combine_fn = optuna_meta_shift_robust(outer_preds_list, sp, n_trials=120)
        sr_meta_all[sp] = full_meta_p
        for fk in range(5):
            vpos = outer_data[fk]["val_bundle"]["v_pos"]
            for pi_loc, pr_i in zip(vpos, nested_sp[fk]):
                nested_cv_preds_sr[fk][pi_loc] = pr_i

        # Full-train 260-model multi-seed inference on test.csv!
        seeds = [2026, 7777]
        cand_ens_t, open_ens_t = None, None
        for sd in seeds:
            c_sd, o_sd = fit_all_shift_robust_models(full_seed_bundles[sd], test_b, sp, hp, seed=sd)
            if cand_ens_t is None:
                cand_ens_t = {k: v / len(seeds) for k, v in c_sd.items()}
                open_ens_t = {k: v / len(seeds) for k, v in o_sd.items()}
            else:
                for k in c_sd:
                    cand_ens_t[k] += c_sd[k] / len(seeds)
                for k in o_sd:
                    open_ens_t[k] += o_sd[k] / len(seeds)

        cn_t, cp_t, ol_t = combine_fn({"bundle": test_b, "cand": cand_ens_t, "open": open_ens_t}, full_meta_p)
        z60_t = test_b["X_open_base"][:, IDX_Z60]
        pi_em = saerens_em_prior(expit(ol_t), PRIOR_PI_NEW[sp])
        print(f"  [{sp.upper()}] Shift-Robust Saerens EM test pi_new = {pi_em:.4f} (KDE prior={PRIOR_PI_NEW[sp]:.3f}, Optuna outer delta={full_meta_p['delta']:+.3f})")

        pr_t = decode_top5_from_probs_v2(cn_t, cp_t, ol_t, delta=full_meta_p["delta"], raw_s1=test_b["raw_s1"], max_sim_inl=test_b["max_sim_inl"], z60=z60_t)
        for loc_i, te_pos in enumerate(test_b["te_pos"]):
            test_preds_sr[te_pos] = " ".join(pr_t[loc_i])
        print(f"Completed Shift-Robust [{sp.upper()}] in {time.time() - ts:.1f}s")

    with open(os.path.join(CACHE_V2_DIR, "optuna_stage2_shift_robust_params.json"), "w") as f:
        json.dump({"stage2_hp": sr_hp_all, "stage3_meta": sr_meta_all}, f, indent=2)

    print("\n================ V2 SHIFT-ROBUST 5-FOLD NESTED CV SUMMARY ================")
    eval_cv_v2(folds, train_df, nested_cv_preds_sr, verbose=True, tag="V2_SHIFT_ROBUST_CHAMPION")

    p_sr = "submissions_v2/submission_v2_shift_robust_champion.csv"
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_sr}).to_csv(p_sr, index=False)
    verify_submission_file(p_sr, test_df, train_df)

    # Also build Hybrid Super-Champion: 70% V2 Shift-Robust + 30% V2 Graph-Enriched on Hyena (where shift was tiny!),
    # and 85% V2 Shift-Robust + 15% V1 (0.82895) Reciprocal-Rank Diversity Super-Blend!
    p_hyb = "submissions_v2/submission_v2_hybrid_grandmaster.csv"
    sub_v2_raw = pd.read_csv("submissions_v2/submission_v2_optuna_champion.csv")
    sub_v1 = pd.read_csv("submissions/submission_best.csv")
    rw = [1.0, 0.46, 0.28, 0.19, 0.14]
    hyb_preds = []
    for i in range(n_te):
        sp = test_df["species"].iat[i]
        sc = {}
        # Hyena had tiny shift -> blend 65% Shift-Robust + 25% Graph-Enriched V2 + 10% V1!
        # Leopard had -0.57 SD graph shift -> blend 80% Shift-Robust + 5% Graph-Enriched V2 + 15% V1!
        w_sr, w_gr, w_v1 = (0.65, 0.25, 0.10) if sp == "hyena" else (0.80, 0.05, 0.15)
        for w_m, row_str in [(w_sr, test_preds_sr[i]), (w_gr, sub_v2_raw["predictions"].iat[i]), (w_v1, sub_v1["predictions"].iat[i])]:
            for r, tok in enumerate(row_str.split(" ")[:5]):
                sc[tok] = sc.get(tok, 0.0) + w_m * rw[r]
        hyb_preds.append(" ".join([k for k, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]]))
    pd.DataFrame({"id": test_df["id"].values, "predictions": hyb_preds}).to_csv(p_hyb, index=False)
    verify_submission_file(p_hyb, test_df, train_df)

    # Compare diagnostics across V1, V2 Raw, V2 Shift-Robust, and V2 Hybrid Grandmaster!
    print("\n================ TEST DIAGNOSTICS COMPARISON ================")
    for lbl, pth in [
        ("V1_Baseline (0.82895 LB)", "submissions/submission_best.csv"),
        ("V2_Graph_Enriched", "submissions_v2/submission_v2_optuna_champion.csv"),
        ("V2_Shift_Robust_Champion", p_sr),
        ("V2_Hybrid_Grandmaster", p_hyb),
    ]:
        sdf = pd.read_csv(pth)
        r1 = sdf["predictions"].str.split(" ").str[0].values
        r2 = sdf["predictions"].str.split(" ").str[1].values
        for sp in ["hyena", "leopard"]:
            m = (test_df["species"].values == sp)
            r1_v1 = sub_v1["predictions"].str.split(" ").str[0].values[m]
            ag = (r1[m] == r1_v1).mean()
            print(f"[{lbl:28s} | {sp:7s}] R1=new:{(r1[m]=='new_individual').mean():.3f} | R2=new:{(r2[m]=='new_individual').mean():.3f} | Top2=new:{((r1[m]=='new_individual')|(r2[m]=='new_individual')).mean():.3f} | agree_V1={ag:.3f}")

    # Check the key sanity test rows (especially te_01952, te_01267, te_00213, te_00314, te_00022)!
    print("\n=== SANITY CHECK ON CRITICAL TEST ROWS ===")
    df_sr = pd.read_csv(p_sr)
    df_hyb = pd.read_csv(p_hyb)
    for tid in ["te_01952", "te_01267", "te_01638", "te_00213", "te_00314", "te_00003", "te_00022", "te_00176"]:
        idx = test_df.index[test_df["id"] == tid][0]
        print(f"{tid} ({test_df.loc[idx, 'species']:7s}):")
        print(f"   V1        : {sub_v1.loc[idx, 'predictions']}")
        print(f"   V2_Graph  : {sub_v2_raw.loc[idx, 'predictions']}")
        print(f"   V2_ShRob  : {df_sr.loc[idx, 'predictions']}")
        print(f"   V2_Hybrid : {df_hyb.loc[idx, 'predictions']}")

    # Set Shift-Robust Champion as root submission.csv (keeping submissions/submission_best.csv untouched!)
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_sr}).to_csv("submission.csv", index=False)
    verify_submission_file("submission.csv", test_df, train_df)
    print(f"\nShift-Robust V2 Pipeline finished in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
