"""
Shared V3 Stage-2 model code used by BOTH `train_and_save_v3.py` (fit + save) and `infer_v3.py` (load + predict).

This is `fit_clean_stage2` from `reproduce_v3_no_optuna.py`, split into a fit half (touches training bundles only)
and a predict half (touches the test bundle only). Model types, seeds, thread counts, score transforms and
blend weights are identical, so a saved-then-loaded ensemble reproduces the original output.
"""
import json
import os

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from cv_v3 import PRIOR_PI_NEW_V3, CACHE_V3_DIR
from reproduce_v3_no_optuna import TOP4_OPEN, clip_logit, row_zscore, build_cand_dataset, get_open_base

CAND_KEYS = ["lgb_rank", "xgb_rank", "cat_rank", "lgb_bin", "xgb_bin", "cat_bin", "hgb_bin", "et_bin"]
RANK_KEYS = ("lgb_rank", "xgb_rank", "cat_rank")
RUN_SPECS = [(2026, 2026), (7777, 7777), (2026, 9999)]  # (subfold split seed, model seed)
PARAM_FILES = {
    "stage2": "optuna_stage2_3_v3_best_params.json",
    "stage3": "v3_final_polished_stage3_params.json",
    "stage1": "optuna_stage1_v3_best_cfg.json",
}


def find_param_file(name):
    """The params live in cache_v3/, or in cache_v3/cache_v3/ when cache_v3.zip was unzipped in place."""
    for d in (CACHE_V3_DIR, os.path.join(CACHE_V3_DIR, "cache_v3")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {CACHE_V3_DIR}/ or {CACHE_V3_DIR}/cache_v3/")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------
# Fit half
# ----------------------------------------------------------------------------------------------------------------
def fit_cand_models(sub_bundles, hs, hp, seed):
    """Fit the 8 candidate rerankers on all subfolds except `hs`."""
    Xtr_r, ytr_r, gtr_r = build_cand_dataset(sub_bundles, hs, for_ranker=True)
    qids_r = np.repeat(np.arange(len(gtr_r)), gtr_r)
    Xtr_b, ytr_b, _ = build_cand_dataset(sub_bundles, hs, for_ranker=False)
    m = {}
    m["lgb_rank"] = lgb.LGBMRanker(objective="lambdarank", eval_at=[1, 5], subsample_freq=1, n_jobs=4, random_state=seed + 11 * hs, verbose=-1, **hp["lgb_rank"]).fit(Xtr_r, ytr_r, group=gtr_r)
    m["xgb_rank"] = xgb.XGBRanker(objective="rank:ndcg", n_jobs=4, random_state=seed + 13 * hs, verbosity=0, **hp["xgb_rank"]).fit(Xtr_r, ytr_r, group=gtr_r)
    m["cat_rank"] = CatBoostRanker(loss_function="QuerySoftMax", thread_count=4, random_seed=seed + 17 * hs, verbose=0, **hp["cat_rank"]).fit(Pool(data=Xtr_r, label=ytr_r.astype(np.float32), group_id=qids_r))
    m["lgb_bin"] = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 19 * hs, verbose=-1, **hp["lgb_bin"]).fit(Xtr_b, ytr_b)
    m["xgb_bin"] = xgb.XGBClassifier(n_jobs=4, random_state=seed + 23 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_bin"]).fit(Xtr_b, ytr_b)
    cr = hp["cat_rank"]
    m["cat_bin"] = CatBoostClassifier(iterations=cr["iterations"], learning_rate=cr["learning_rate"], depth=cr["depth"], l2_leaf_reg=cr["l2_leaf_reg"], thread_count=4, random_seed=seed + 29 * hs, verbose=0).fit(Xtr_b, ytr_b)
    m["hgb_bin"] = HistGradientBoostingClassifier(random_state=seed + 31 * hs, **hp["hgb_bin"]).fit(Xtr_b, ytr_b)
    m["et_bin"] = ExtraTreesClassifier(n_estimators=180, max_depth=14, min_samples_leaf=8, max_features="sqrt", n_jobs=4, random_state=seed + 33 * hs).fit(Xtr_b, ytr_b)
    return m


def fit_open_models(X_tr, y_tr, w_tr, hp, seed, hs):
    """Fit the 4 open-set ('new_individual') detectors."""
    return {
        "lgb_open": lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 43 * hs, verbose=-1, **hp["lgb_open"]).fit(X_tr, y_tr, sample_weight=w_tr),
        "xgb_open": xgb.XGBClassifier(n_jobs=4, random_state=seed + 47 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_open"]).fit(X_tr, y_tr, sample_weight=w_tr),
        "cat_open": CatBoostClassifier(thread_count=4, random_seed=seed + 53 * hs, verbose=0, **hp["cat_open"]).fit(X_tr, y_tr, sample_weight=w_tr),
        "hgb_open": HistGradientBoostingClassifier(random_state=seed + 59 * hs, **hp["hgb_open"]).fit(X_tr, y_tr, sample_weight=w_tr),
    }


def fit_stage2_run(sub_bundles, species, hp, seed):
    """One (split seed, model seed) run: n_sf x 8 candidate models + n_sf x 4 open-set models."""
    pi_new = PRIOR_PI_NEW_V3[species]
    n_sf = len(sub_bundles)
    cand_models, oof = [], []
    for hs in range(n_sf):
        m = fit_cand_models(sub_bundles, hs, hp, seed)
        cand_models.append(m)
        hb = sub_bundles[hs]
        nh, Kh, Fh = hb["X_cand"].shape
        oof.append(cand_scores(m, hb["X_cand"].reshape(-1, Fh), nh, Kh))

    sub_open = [make_stacked(sub_bundles[s], species, blend_bin(oof[s]), blend_rank(oof[s])) for s in range(n_sf)]

    open_models = []
    for hs in range(n_sf):
        Xo_l, yo_l, wo_l = [], [], []
        for idx_s, sb in enumerate(sub_bundles):
            if idx_s == hs:
                continue
            isn = sb["is_new"]
            Xo_l.append(sub_open[idx_s])
            yo_l.append(isn.astype(np.int32))
            nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
            wo_l.append(np.where(isn, pi_new / nn, (1.0 - pi_new) / nk) * len(isn))
        open_models.append(fit_open_models(np.vstack(Xo_l), np.concatenate(yo_l), np.concatenate(wo_l), hp, seed, hs))
    return {"cand": cand_models, "open": open_models}


# ----------------------------------------------------------------------------------------------------------------
# Predict half
# ----------------------------------------------------------------------------------------------------------------
def cand_scores(models, X_flat, n, K):
    """Rankers -> row z-scored scores; binary classifiers -> clipped logits. Shape (n, K) each."""
    out = {}
    for k in CAND_KEYS:
        if k in RANK_KEYS:
            out[k] = row_zscore(models[k].predict(X_flat).reshape(n, K))
        else:
            out[k] = clip_logit(models[k].predict_proba(X_flat)[:, 1]).reshape(n, K)
    return out


def blend_bin(c):
    return 0.27 * c["hgb_bin"] + 0.25 * c["cat_bin"] + 0.24 * c["lgb_bin"] + 0.19 * c["xgb_bin"] + 0.05 * c["et_bin"]


def blend_rank(c):
    return (c["cat_rank"] + c["lgb_rank"] + c["xgb_rank"]) / 3.0


def make_stacked(bundle, species, bin_m, rnk_m):
    X_ob = get_open_base(bundle, species)
    sb, sr = -np.sort(-bin_m, axis=1), -np.sort(-rnk_m, axis=1)
    pb, pr = softmax(bin_m, axis=1), softmax(1.5 * rnk_m, axis=1)
    eb = -np.sum(pb * np.log(np.clip(pb, 1e-12, None)), axis=1)
    er = -np.sum(pr * np.log(np.clip(pr, 1e-12, None)), axis=1)
    return np.column_stack([X_ob, sb[:, 0], sb[:, 0] - sb[:, 1], pb.max(1), eb, sr[:, 0], sr[:, 0] - sr[:, 1], pr.max(1), er]).astype(np.float32)


def predict_stage2_run(run, target_bundle, species):
    """Apply one saved run to a target bundle -> (candidate score dict, open-set logit dict)."""
    n_sf = len(run["cand"])
    nt, Kt, Ft = target_bundle["X_cand"].shape
    X_t_c = target_bundle["X_cand"].reshape(-1, Ft)
    tgt_c = {k: np.zeros((nt, Kt), dtype=np.float64) for k in CAND_KEYS}
    for m in run["cand"]:
        s = cand_scores(m, X_t_c, nt, Kt)
        for k in CAND_KEYS:
            tgt_c[k] += s[k] / n_sf

    X_t_open = make_stacked(target_bundle, species, blend_bin(tgt_c), blend_rank(tgt_c))
    tgt_o = {k: np.zeros(nt, dtype=np.float64) for k in TOP4_OPEN}
    for om in run["open"]:
        for k in TOP4_OPEN:
            tgt_o[k] += clip_logit(om[k].predict_proba(X_t_open)[:, 1]) / n_sf
    return tgt_c, tgt_o
