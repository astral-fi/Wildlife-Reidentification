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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker, Pool

from cv_v2 import build_or_load_cv_v2, eval_cv_v2, ap_at_5, PRIOR_PI_NEW, TEST_SPECIES_WEIGHTS, DATA_DIR, CACHE_V2_DIR

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def clip_logit(p):
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def row_zscore(M):
    return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-4)


def verify_submission_file(sub_path, test_df, train_df):
    sub = pd.read_csv(sub_path)
    assert len(sub) == len(test_df) == 2791, f"Bad row count: {len(sub)}"
    assert list(sub.columns) == ["id", "predictions"], f"Bad cols: {list(sub.columns)}"
    assert (sub["id"].values == test_df["id"].values).all(), "ID mismatch!"
    assert sub["predictions"].isna().sum() == 0, "NaNs found!"
    valid_by_sp = {
        sp: set(train_df[train_df["species"] == sp]["identity"].unique()) | {"new_individual"}
        for sp in ["hyena", "leopard"]
    }
    for i, (ps, sp) in enumerate(zip(sub["predictions"].values, test_df["species"].values)):
        toks = ps.strip().split(" ")
        assert 1 <= len(toks) <= 5 and len(set(toks)) == len(toks), f"Row {i} invalid tokens: {toks}"
        for t in toks:
            assert t in valid_by_sp[sp], f"Row {i} wrong species token {t} for {sp}"
    print(f"[VERIFIED PASS] {sub_path}: 2,791 rows, 0 NaNs, exact unique same-species Top-5!")


def known_map5_from_scores(cand_names, scores_2d, y_true, is_new):
    kn = ~is_new
    cn = cand_names[kn]
    sc = scores_2d[kn]
    yt = y_true[kn]
    ord5 = np.argsort(-sc, axis=1)[:, :5]
    top5 = np.take_along_axis(cn, ord5, axis=1)
    aps = np.zeros(len(yt), dtype=np.float64)
    for r in range(5):
        hit = (top5[:, r] == yt) & (aps == 0.0)
        aps[hit] = 1.0 / (r + 1)
    return float(aps.mean())


def open_map5_from_probs(cand_names, cand_probs, open_logits, y_true, is_new, pi_new, delta=0.0, raw_s1=None, max_sim_inl=None):
    """Exact vectorized prior-weighted MAP@5 evaluation directly from calibrated candidate probabilities!"""
    w = np.where(is_new, pi_new / max(1, int(is_new.sum())), (1.0 - pi_new) / max(1, int((~is_new).sum())))
    ord5 = np.argsort(-cand_probs, axis=1)[:, :5]
    top5_names = np.take_along_axis(cand_names, ord5, axis=1)
    cp5 = np.take_along_axis(cand_probs, ord5, axis=1)

    p_new = expit(open_logits + delta)
    if raw_s1 is not None and max_sim_inl is not None:
        ultra = (raw_s1 >= 0.78) & (max_sim_inl >= 12.0)
        p_new = np.where(ultra, np.minimum(p_new, 0.12), p_new)

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


def decode_top5_from_probs(cand_names, cand_probs, open_logits, delta=0.0, raw_s1=None, max_sim_inl=None):
    ord5 = np.argsort(-cand_probs, axis=1)[:, :5]
    top5_names = np.take_along_axis(cand_names, ord5, axis=1)
    cp5 = np.take_along_axis(cand_probs, ord5, axis=1)
    p_new = expit(open_logits + delta)
    if raw_s1 is not None and max_sim_inl is not None:
        ultra = (raw_s1 >= 0.78) & (max_sim_inl >= 12.0)
        p_new = np.where(ultra, np.minimum(p_new, 0.12), p_new)

    preds = []
    for i in range(len(cand_names)):
        pn = float(p_new[i])
        kp = (1.0 - pn) * cp5[i]
        items = [("new_individual", pn)] + [(top5_names[i, r], float(kp[r])) for r in range(5)]
        items.sort(key=lambda x: -x[1])
        preds.append([x[0] for x in items[:5]])
    return preds


def decode_top5(cand_names, cand_scores, open_logits, gamma=1.0, delta=0.0, raw_s1=None, max_sim_inl=None):
    return decode_top5_from_probs(
        cand_names, softmax(gamma * cand_scores, axis=1), open_logits, delta=delta, raw_s1=raw_s1, max_sim_inl=max_sim_inl
    )


def precompute_graph_struct(W_qq, w_qq_sum, has_qq, q_comp, nc_q):
    """Precomputes neighbor lists & connected components once per bundle for 10x faster Optuna Stage C!"""
    nbrs_list = []
    ws_list = []
    for i in range(len(has_qq)):
        if has_qq[i]:
            nb = np.where(W_qq[i] > 0)[0]
            nbrs_list.append(nb)
            ws_list.append((W_qq[i, nb] / w_qq_sum[i, 0]).astype(np.float64))
        else:
            nbrs_list.append(None)
            ws_list.append(None)
    comp_members = {}
    for cid in range(nc_q):
        idx_c = np.where(q_comp == cid)[0]
        if len(idx_c) >= 2:
            comp_members[cid] = idx_c
    return nbrs_list, ws_list, comp_members


def apply_two_level_graph_postproc(
    cand_names, cand_scores, open_logits, W_qq, w_qq_sum, has_qq, q_comp, nc_q,
    gamma=2.0, alpha_open_1h=0.25, alpha_cand_1h=0.20, alpha_open_cc=0.18, alpha_cand_cc=0.12,
    graph_struct=None
):
    """
    Temperature-coherent Two-level Transductive Graph Posterior Consolidation:
      1) Applies `softmax(gamma * cand_scores)` FIRST so probabilities are sharp & unified across all rows!
      2) 1-hop similarity-weighted smoothing (W_qq) on open-set logits & candidate posteriors
      3) Multi-hop Connected-Component Cluster-Level pooling (q_comp)
    Returns `(cn_out, probs_out, open_out)` where `probs_out` is ready for direct Top-5 Bayes decoding!
    """
    if graph_struct is None:
        nbrs_list, ws_list, comp_members = precompute_graph_struct(W_qq, w_qq_sum, has_qq, q_comp, nc_q)
    else:
        nbrs_list, ws_list, comp_members = graph_struct

    open_out = open_logits.copy()
    if np.any(has_qq) and alpha_open_1h > 1e-4:
        nbr_o = (W_qq[has_qq] @ open_logits) / w_qq_sum[has_qq, 0]
        open_out[has_qq] = (1.0 - alpha_open_1h) * open_out[has_qq] + alpha_open_1h * nbr_o

    if alpha_open_cc > 1e-4 and len(comp_members) > 0:
        cc_o = open_out.copy()
        for cid, idx_c in comp_members.items():
            c_mean = float(open_out[idx_c].mean())
            c_min = float(open_out[idx_c].min())
            c_target = 0.65 * c_mean + 0.35 * c_min
            cc_o[idx_c] = (1.0 - alpha_open_cc) * open_out[idx_c] + alpha_open_cc * c_target
        open_out = cc_o

    # Temperature-calibrated candidate probabilities FIRST (unified for isolated & clustered rows!)
    probs = softmax(gamma * cand_scores, axis=1)
    cn_out = cand_names.copy()
    cp_out = probs.copy()
    K = cand_names.shape[1]

    if (np.any(has_qq) and alpha_cand_1h > 1e-4) or (alpha_cand_cc > 1e-4 and len(comp_members) > 0):
        # Pre-aggregate per-component probability dictionaries once (O(sum |C|) instead of O(sum |C|^2)!)
        comp_prob_map = {}
        if alpha_cand_cc > 1e-4:
            for cid, idx_c in comp_members.items():
                cmap = {}
                inv_sz = 1.0 / float(len(idx_c))
                for nj in idx_c:
                    cn_j = cand_names[nj]
                    pr_j = probs[nj]
                    for r in range(K):
                        c = cn_j[r]
                        cmap[c] = cmap.get(c, 0.0) + inv_sz * pr_j[r]
                comp_prob_map[cid] = cmap

        for i in range(len(cand_names)):
            do_1h = has_qq[i] and alpha_cand_1h > 1e-4
            cid_i = q_comp[i]
            do_cc = (cid_i in comp_prob_map) and alpha_cand_cc > 1e-4
            if not (do_1h or do_cc):
                continue
            w_self = 1.0 - (alpha_cand_1h if do_1h else 0.0) - (alpha_cand_cc if do_cc else 0.0)
            smap = {}
            cn_i = cand_names[i]
            pr_i = probs[i]
            for r in range(K):
                c = cn_i[r]
                smap[c] = smap.get(c, 0.0) + w_self * pr_i[r]
            if do_1h:
                for nj, wj in zip(nbrs_list[i], ws_list[i]):
                    cn_j = cand_names[nj]
                    pr_j = probs[nj]
                    a_w = alpha_cand_1h * wj
                    for r in range(K):
                        c = cn_j[r]
                        smap[c] = smap.get(c, 0.0) + a_w * pr_j[r]
            if do_cc:
                for c, pv in comp_prob_map[cid_i].items():
                    smap[c] = smap.get(c, 0.0) + alpha_cand_cc * pv
            top_k_it = sorted(smap.items(), key=lambda x: -x[1])[:K]
            for r, (c, pr) in enumerate(top_k_it):
                cn_out[i, r] = c
                cp_out[i, r] = pr

    return cn_out, cp_out, open_out


def saerens_em_prior(probs_test, pi_train, max_iter=50, tol=1e-5):
    """
    Saerens et al. (2002) EM algorithm estimating the true target prior pi_test
    purely from calibrated classifier probabilities on unlabeled test.csv!
    """
    p = np.clip(probs_test, 1e-5, 1.0 - 1e-5)
    pi_t = float(pi_train)
    for _ in range(max_iter):
        num = (pi_t / pi_train) * p
        den = num + ((1.0 - pi_t) / (1.0 - pi_train)) * (1.0 - p)
        pi_new = float(np.mean(num / den))
        if abs(pi_new - pi_t) < tol:
            pi_t = pi_new
            break
        pi_t = pi_new
    return pi_t


# ---------------------------------------------------------------------------
# Optuna Stage B: Tune Listwise Rankers + Pointwise Rerankers + Open-Set Models!
# ---------------------------------------------------------------------------

def build_cand_dataset(sub_bundles, hold_s, for_ranker=False):
    X_l, y_l, g_l = [], [], []
    for idx_s, sb in enumerate(sub_bundles):
        if idx_s == hold_s:
            continue
        nq, K, F = sb["X_cand"].shape
        ym = (sb["cand_names"] == sb["y_true"][:, None]).astype(np.int32)
        if for_ranker:
            # Listwise rankers train strictly on KNOWN queries that contain the true class in Top-K!
            keep = (~sb["is_new"]) & (ym.sum(axis=1) > 0)
            n_keep = int(keep.sum())
            X_l.append(sb["X_cand"][keep].reshape(-1, F))
            y_l.append(ym[keep].ravel())
            g_l.extend([K] * n_keep)
        else:
            keep = (~sb["is_new"]) | ((np.arange(nq) % 4) == 0)
            X_l.append(sb["X_cand"][keep].reshape(-1, F))
            y_l.append(ym[keep].ravel())
    X_tr = np.vstack(X_l)
    y_tr = np.concatenate(y_l)
    return X_tr, y_tr, np.array(g_l, dtype=np.int32)


def optuna_tune_stage2_hyperparams(species, full_sub_bundles, n_trials_tree=18):
    """
    Runs real Optuna TPE HPO per species on the 5 full-train internal sub-folds (zero test leakage!)
    to tune:
      1) LGBMRanker (lambdarank)
      2) XGBRanker (rank:ndcg)
      3) CatBoostRanker (QuerySoftMax)
      4) LGBMClassifier, XGBClassifier, CatBoostClassifier (candidate binary)
      5) LGBMClassifier, XGBClassifier, CatBoostClassifier (open-set new_individual)
    """
    pi_new = PRIOR_PI_NEW[species]
    print(f"\n================ OPTUNA STAGE B: STAGE-2 HPO [{species.upper()}] ================")
    best_hp = {}

    # Pre-split 3 fast validation folds (0, 1, 2) out of 5 sub-folds for brisk, low-variance HPO
    eval_subs = [0, 1, 2]

    # 1. Optuna for LGBMRanker (LambdaRank)
    def obj_lgb_rank(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 140, 360, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 60),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.2, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
        }
        scs = []
        for hs in eval_subs:
            Xtr, ytr, gtr = build_cand_dataset(full_sub_bundles, hs, for_ranker=True)
            hb = full_sub_bundles[hs]
            nh, Kh, Fh = hb["X_cand"].shape
            m = lgb.LGBMRanker(objective="lambdarank", eval_at=[1, 5], subsample_freq=1, n_jobs=4, random_state=42 + hs, verbose=-1, **p)
            m.fit(Xtr, ytr, group=gtr)
            pr = m.predict(hb["X_cand"].reshape(-1, Fh)).reshape(nh, Kh)
            scs.append(known_map5_from_scores(hb["cand_names"], pr, hb["y_true"], hb["is_new"]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(obj_lgb_rank, n_trials=n_trials_tree)
    best_hp["lgb_rank"] = st.best_params
    print(f"  [{species}] Optuna LGBMRanker (LambdaRank) Best Known MAP@5 = {st.best_value:.5f}")

    # 2. Optuna for XGBRanker (rank:ndcg)
    def obj_xgb_rank(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 140, 340, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.2, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
        }
        scs = []
        for hs in eval_subs:
            Xtr, ytr, gtr = build_cand_dataset(full_sub_bundles, hs, for_ranker=True)
            hb = full_sub_bundles[hs]
            nh, Kh, Fh = hb["X_cand"].shape
            m = xgb.XGBRanker(objective="rank:ndcg", n_jobs=4, random_state=42 + hs, verbosity=0, **p)
            m.fit(Xtr, ytr, group=gtr)
            pr = m.predict(hb["X_cand"].reshape(-1, Fh)).reshape(nh, Kh)
            scs.append(known_map5_from_scores(hb["cand_names"], pr, hb["y_true"], hb["is_new"]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43))
    st.optimize(obj_xgb_rank, n_trials=n_trials_tree)
    best_hp["xgb_rank"] = st.best_params
    print(f"  [{species}] Optuna XGBRanker (rank:ndcg)  Best Known MAP@5 = {st.best_value:.5f}")

    # 3. Optuna for CatBoostRanker (QuerySoftMax!)
    def obj_cat_rank(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 180, 380, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 18.0, log=True),
        }
        scs = []
        for hs in eval_subs[:2]:
            Xtr, ytr, gtr = build_cand_dataset(full_sub_bundles, hs, for_ranker=True)
            qids = np.repeat(np.arange(len(gtr)), gtr)
            hb = full_sub_bundles[hs]
            nh, Kh, Fh = hb["X_cand"].shape
            pool_tr = Pool(data=Xtr, label=ytr.astype(np.float32), group_id=qids)
            m = CatBoostRanker(loss_function="QuerySoftMax", thread_count=4, random_seed=42 + hs, verbose=0, **p)
            m.fit(pool_tr)
            pr = m.predict(hb["X_cand"].reshape(-1, Fh)).reshape(nh, Kh)
            scs.append(known_map5_from_scores(hb["cand_names"], pr, hb["y_true"], hb["is_new"]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=44))
    st.optimize(obj_cat_rank, n_trials=max(10, n_trials_tree // 2))
    best_hp["cat_rank"] = st.best_params
    print(f"  [{species}] Optuna CatBoost (QuerySoftMax) Best Known MAP@5 = {st.best_value:.5f}")

    # 4. Optuna for Pointwise Binary LGBM + XGB + Cat Candidate Models
    def obj_lgb_bin(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 160, 360, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 9, 28),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 18, 60),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.3, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.5, 18.0, log=True),
        }
        scs = []
        for hs in eval_subs:
            Xtr, ytr, _ = build_cand_dataset(full_sub_bundles, hs, for_ranker=False)
            hb = full_sub_bundles[hs]
            nh, Kh, Fh = hb["X_cand"].shape
            m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=42 + hs, verbose=-1, **p)
            m.fit(Xtr, ytr)
            pr = clip_logit(m.predict_proba(hb["X_cand"].reshape(-1, Fh))[:, 1]).reshape(nh, Kh)
            scs.append(known_map5_from_scores(hb["cand_names"], pr, hb["y_true"], hb["is_new"]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=45))
    st.optimize(obj_lgb_bin, n_trials=n_trials_tree)
    best_hp["lgb_bin"] = st.best_params
    print(f"  [{species}] Optuna LGBMClassifier (Cand)  Best Known MAP@5 = {st.best_value:.5f}")

    def obj_xgb_bin(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 160, 340, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 3.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 0.85),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.3, 8.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.5, 18.0, log=True),
        }
        scs = []
        for hs in eval_subs:
            Xtr, ytr, _ = build_cand_dataset(full_sub_bundles, hs, for_ranker=False)
            hb = full_sub_bundles[hs]
            nh, Kh, Fh = hb["X_cand"].shape
            m = xgb.XGBClassifier(n_jobs=4, random_state=42 + hs, eval_metric="logloss", verbosity=0, **p)
            m.fit(Xtr, ytr)
            pr = clip_logit(m.predict_proba(hb["X_cand"].reshape(-1, Fh))[:, 1]).reshape(nh, Kh)
            scs.append(known_map5_from_scores(hb["cand_names"], pr, hb["y_true"], hb["is_new"]))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=49))
    st.optimize(obj_xgb_bin, n_trials=max(12, n_trials_tree - 2))
    best_hp["xgb_bin"] = st.best_params
    print(f"  [{species}] Optuna XGBClassifier (Cand)   Best Known MAP@5 = {st.best_value:.5f}")

    # 5. Optuna for Open-Set Detectors (LGBM, XGB, CatBoost) optimizing calibrated ROC-AUC - 0.35 * Weighted LogLoss!
    from sklearn.metrics import roc_auc_score, log_loss
    def build_open_tr(hs):
        Xo, yo, wo = [], [], []
        for idx_s, sb in enumerate(full_sub_bundles):
            if idx_s == hs:
                continue
            isn = sb["is_new"]
            Xo.append(sb["X_open_base"])
            yo.append(isn.astype(np.int32))
            nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
            wo.append(np.where(isn, pi_new / nn, (1.0 - pi_new) / nk) * len(isn))
        return np.vstack(Xo), np.concatenate(yo), np.concatenate(wo)

    def eval_open_cal(hb, pr):
        isn = hb["is_new"]
        y_i = isn.astype(int)
        nn, nk = max(1, int(isn.sum())), max(1, int((~isn).sum()))
        w_i = np.where(isn, pi_new / nn, (1.0 - pi_new) / nk)
        auc = roc_auc_score(y_i, pr)
        ll = log_loss(y_i, np.clip(pr, 1e-5, 1.0 - 1e-5), sample_weight=w_i)
        return float(auc - 0.35 * ll)

    def obj_lgb_open(trial):
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
            hb = full_sub_bundles[hs]
            m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=42 + hs, verbose=-1, **p)
            m.fit(Xo, yo, sample_weight=wo)
            pr = m.predict_proba(hb["X_open_base"])[:, 1]
            scs.append(eval_open_cal(hb, pr))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=46))
    st.optimize(obj_lgb_open, n_trials=n_trials_tree)
    best_hp["lgb_open"] = st.best_params
    print(f"  [{species}] Optuna LGBM Open-Set        Best Cal-Obj     = {st.best_value:.5f}")

    def obj_xgb_open(trial):
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
            hb = full_sub_bundles[hs]
            m = xgb.XGBClassifier(n_jobs=4, random_state=42 + hs, eval_metric="logloss", verbosity=0, **p)
            m.fit(Xo, yo, sample_weight=wo)
            pr = m.predict_proba(hb["X_open_base"])[:, 1]
            scs.append(eval_open_cal(hb, pr))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=47))
    st.optimize(obj_xgb_open, n_trials=n_trials_tree)
    best_hp["xgb_open"] = st.best_params
    print(f"  [{species}] Optuna XGB Open-Set         Best Cal-Obj     = {st.best_value:.5f}")

    def obj_cat_open(trial):
        p = {
            "iterations": trial.suggest_int("iterations", 180, 360, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.07, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 20.0, log=True),
        }
        scs = []
        for hs in eval_subs[:2]:
            Xo, yo, wo = build_open_tr(hs)
            hb = full_sub_bundles[hs]
            m = CatBoostClassifier(thread_count=4, random_seed=42 + hs, verbose=0, **p)
            m.fit(Xo, yo, sample_weight=wo)
            pr = m.predict_proba(hb["X_open_base"])[:, 1]
            scs.append(eval_open_cal(hb, pr))
        return float(np.mean(scs))

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=48))
    st.optimize(obj_cat_open, n_trials=max(10, n_trials_tree // 2))
    best_hp["cat_open"] = st.best_params
    print(f"  [{species}] Optuna CatBoost Open-Set    Best Cal-Obj     = {st.best_value:.5f}")

    return best_hp


def fit_all_stage2_models_on_subfolds(sub_bundles, target_bundle, species, hp, seed=42):
    """
    Fits ALL 8 Candidate Rankers (3 Listwise + 5 Pointwise) & ALL 5 Open-Set Detectors across `sub_bundles`!
    Returns standardized raw per-model predictions on `target_bundle` (for Optuna Stage C & final ensembling!).
    """
    pi_new = PRIOR_PI_NEW[species]
    n_sf = len(sub_bundles)
    cand_models_keys = ["lgb_rank", "xgb_rank", "cat_rank", "lgb_bin", "xgb_bin", "cat_bin", "mlp_bin", "logreg_bin"]
    open_models_keys = ["lgb_open", "xgb_open", "cat_open", "mlp_open", "logreg_open"]

    nt, Kt, Ft = target_bundle["X_cand"].shape
    X_t_c = target_bundle["X_cand"].reshape(-1, Ft)

    oof_c = {k: [] for k in cand_models_keys}
    tgt_c = {k: np.zeros((nt, Kt), dtype=np.float64) for k in cand_models_keys}

    for hs in range(n_sf):
        hb = sub_bundles[hs]
        nh, Kh, Fh = hb["X_cand"].shape
        X_h_c = hb["X_cand"].reshape(-1, Fh)

        # Listwise training set
        Xtr_r, ytr_r, gtr_r = build_cand_dataset(sub_bundles, hs, for_ranker=True)
        qids_r = np.repeat(np.arange(len(gtr_r)), gtr_r)

        # Pointwise binary training set
        Xtr_b, ytr_b, _ = build_cand_dataset(sub_bundles, hs, for_ranker=False)

        # 1) LGBMRanker
        m = lgb.LGBMRanker(objective="lambdarank", eval_at=[1, 5], subsample_freq=1, n_jobs=4, random_state=seed + 11 * hs, verbose=-1, **hp["lgb_rank"])
        m.fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["lgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["lgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        # 2) XGBRanker
        m = xgb.XGBRanker(objective="rank:ndcg", n_jobs=4, random_state=seed + 13 * hs, verbosity=0, **hp["xgb_rank"])
        m.fit(Xtr_r, ytr_r, group=gtr_r)
        oof_c["xgb_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["xgb_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        # 3) CatBoostRanker (QuerySoftMax)
        m = CatBoostRanker(loss_function="QuerySoftMax", thread_count=4, random_seed=seed + 17 * hs, verbose=0, **hp["cat_rank"])
        m.fit(Pool(data=Xtr_r, label=ytr_r.astype(np.float32), group_id=qids_r))
        oof_c["cat_rank"].append(row_zscore(m.predict(X_h_c).reshape(nh, Kh)))
        tgt_c["cat_rank"] += row_zscore(m.predict(X_t_c).reshape(nt, Kt)) / n_sf

        # 4) LGBMClassifier
        m = lgb.LGBMClassifier(subsample_freq=1, n_jobs=4, random_state=seed + 19 * hs, verbose=-1, **hp["lgb_bin"])
        m.fit(Xtr_b, ytr_b)
        oof_c["lgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["lgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        # 5) XGBClassifier (with its own Optuna-tuned binary hyperparameters!)
        m = xgb.XGBClassifier(n_jobs=4, random_state=seed + 23 * hs, eval_metric="logloss", verbosity=0, **hp["xgb_bin"])
        m.fit(Xtr_b, ytr_b)
        oof_c["xgb_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["xgb_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        # 6) CatBoostClassifier
        m = CatBoostClassifier(
            iterations=hp["cat_rank"]["iterations"], learning_rate=hp["cat_rank"]["learning_rate"],
            depth=hp["cat_rank"]["depth"], l2_leaf_reg=hp["cat_rank"]["l2_leaf_reg"],
            thread_count=4, random_seed=seed + 29 * hs, verbose=0
        )
        m.fit(Xtr_b, ytr_b)
        oof_c["cat_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["cat_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        # 7) MLPClassifier (from scratch)
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=512, alpha=2.5, max_iter=85, early_stopping=True, random_state=seed + 31 * hs))
        m.fit(Xtr_b, ytr_b)
        oof_c["mlp_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["mlp_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

        # 8) LogisticRegression
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.35, max_iter=300, random_state=seed + 37 * hs))
        m.fit(Xtr_b, ytr_b)
        oof_c["logreg_bin"].append(clip_logit(m.predict_proba(X_h_c)[:, 1]).reshape(nh, Kh))
        tgt_c["logreg_bin"] += clip_logit(m.predict_proba(X_t_c)[:, 1]).reshape(nt, Kt) / n_sf

    # Stack BOTH Listwise Ranker & Pointwise Classifier confidence into X_open_stacked (+8 features -> 105 features!)
    def make_stacked_open(X_base, bin_mat, rank_mat):
        sb = -np.sort(-bin_mat, axis=1)
        sr = -np.sort(-rank_mat, axis=1)
        pb = softmax(bin_mat, axis=1)
        pr = softmax(1.5 * rank_mat, axis=1)
        eb = -np.sum(pb * np.log(np.clip(pb, 1e-12, None)), axis=1)
        er = -np.sum(pr * np.log(np.clip(pr, 1e-12, None)), axis=1)
        return np.column_stack([X_base, sb[:, 0], sb[:, 0] - sb[:, 1], pb.max(1), eb, sr[:, 0], sr[:, 0] - sr[:, 1], pr.max(1), er]).astype(np.float32)

    sub_open_st = []
    for s in range(n_sf):
        b_avg = 0.28 * oof_c["lgb_bin"][s] + 0.28 * oof_c["xgb_bin"][s] + 0.24 * oof_c["cat_bin"][s] + 0.10 * oof_c["mlp_bin"][s] + 0.10 * oof_c["logreg_bin"][s]
        r_avg = (oof_c["lgb_rank"][s] + oof_c["xgb_rank"][s] + oof_c["cat_rank"][s]) / 3.0
        sub_open_st.append(make_stacked_open(sub_bundles[s]["X_open_base"], b_avg, r_avg))

    t_b_avg = 0.28 * tgt_c["lgb_bin"] + 0.28 * tgt_c["xgb_bin"] + 0.24 * tgt_c["cat_bin"] + 0.10 * tgt_c["mlp_bin"] + 0.10 * tgt_c["logreg_bin"]
    t_r_avg = (tgt_c["lgb_rank"] + tgt_c["xgb_rank"] + tgt_c["cat_rank"]) / 3.0
    X_t_open = make_stacked_open(target_bundle["X_open_base"], t_b_avg, t_r_avg)

    tgt_o = {k: np.zeros(nt, dtype=np.float64) for k in open_models_keys}

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

    # Also include Stage-1 standardized score in tgt_c
    tgt_c["stage1"] = row_zscore(target_bundle["stage1_top_scores"])
    return tgt_c, tgt_o


def optuna_meta_blend_and_postproc(outer_preds_list, species, n_trials=90):
    """
    Optuna Stage C: Nested Leave-One-Outer-Fold-Out Meta-Optimizer!
    Tunes exact ensemble weights across all 9 candidate rankers + 5 open-set detectors +
    2-level graph cluster post-processing + (gamma, delta) strictly on the 4 OTHER outer folds!
    """
    pi_new = PRIOR_PI_NEW[species]
    ckeys = ["stage1", "lgb_rank", "xgb_rank", "cat_rank", "lgb_bin", "xgb_bin", "cat_bin", "mlp_bin", "logreg_bin"]
    okeys = ["lgb_open", "xgb_open", "cat_open", "mlp_open", "logreg_open"]

    # Precompute graph adjacency lists once for 10x faster Stage C Optuna search!
    for item in outer_preds_list:
        b = item["bundle"]
        item["graph_struct"] = precompute_graph_struct(b["W_qq"], b["w_qq_sum"], b["has_qq"], b["q_comp"], b["nc_q"])

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
        cn, cp, ol = apply_two_level_graph_postproc(
            b["cand_names"], c_tot, o_tot, b["W_qq"], b["w_qq_sum"], b["has_qq"], b["q_comp"], b["nc_q"],
            gamma=p["gamma"], alpha_open_1h=p["a_o1"], alpha_cand_1h=p["a_c1"],
            alpha_open_cc=p["a_oc"], alpha_cand_cc=p["a_cc"],
            graph_struct=fd_item.get("graph_struct")
        )
        return cn, cp, ol

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
            "mix_rank": trial.suggest_float("mix_rank", 0.10, 0.75),
            "wo_lgb": trial.suggest_float("wo_lgb", 0.10, 0.50),
            "wo_xgb": trial.suggest_float("wo_xgb", 0.10, 0.50),
            "wo_cat": trial.suggest_float("wo_cat", 0.10, 0.50),
            "wo_mlp": trial.suggest_float("wo_mlp", 0.0, 0.25),
            "wo_lr": trial.suggest_float("wo_lr", 0.0, 0.20),
            "a_o1": trial.suggest_float("a_o1", 0.0, 0.45),
            "a_c1": trial.suggest_float("a_c1", 0.0, 0.40),
            "a_oc": trial.suggest_float("a_oc", 0.0, 0.35),
            "a_cc": trial.suggest_float("a_cc", 0.0, 0.30),
            "gamma": trial.suggest_float("gamma", 1.2, 4.5),
            "delta": trial.suggest_float("delta", -0.80, 1.10),
        }

    base_meta = {
        "w_stage1": 0.12, "w_lgb_rank": 0.30, "w_xgb_rank": 0.30, "w_cat_rank": 0.30,
        "w_lgb_bin": 0.28, "w_xgb_bin": 0.28, "w_cat_bin": 0.24, "w_mlp_bin": 0.10, "w_logreg_bin": 0.10,
        "mix_rank": 0.45,
        "wo_lgb": 0.30, "wo_xgb": 0.30, "wo_cat": 0.25, "wo_mlp": 0.08, "wo_lr": 0.07,
        "a_o1": 0.22, "a_c1": 0.18, "a_oc": 0.12, "a_cc": 0.08,
        "gamma": 2.3, "delta": 0.0,
    }

    # Nested LOFO evaluation across all 5 outer folds!
    nested_preds_by_fold = {}
    learned_params_per_fold = []

    for hold_f in range(5):
        train_fs = [outer_preds_list[f] for f in range(5) if f != hold_f]

        def obj_meta(trial):
            p = suggest_meta_params(trial)
            scs = []
            for item in train_fs:
                cn, cp, ol = combine_fold(item, p)
                b = item["bundle"]
                scs.append(open_map5_from_probs(cn, cp, ol, b["y_true"], b["is_new"], pi_new, delta=p["delta"], raw_s1=b["raw_s1"], max_sim_inl=b["max_sim_inl"]))
            return float(np.mean(scs))

        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=100 + hold_f))
        st.enqueue_trial(base_meta)
        st.optimize(obj_meta, n_trials=n_trials)
        bp = st.best_params
        learned_params_per_fold.append(bp)

        val_item = outer_preds_list[hold_f]
        cn_v, cp_v, ol_v = combine_fold(val_item, bp)
        bv = val_item["bundle"]
        nested_preds_by_fold[hold_f] = decode_top5_from_probs(cn_v, cp_v, ol_v, delta=bp["delta"], raw_s1=bv["raw_s1"], max_sim_inl=bv["max_sim_inl"])

    # Also fit one full-5-fold meta param dict for test.csv inference!
    def obj_full(trial):
        p = suggest_meta_params(trial)
        return float(np.mean([
            open_map5_from_probs(*combine_fold(item, p), item["bundle"]["y_true"], item["bundle"]["is_new"], pi_new, delta=p["delta"], raw_s1=item["bundle"]["raw_s1"], max_sim_inl=item["bundle"]["max_sim_inl"])
            for item in outer_preds_list
        ]))

    st_full = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026))
    st_full.enqueue_trial(base_meta)
    st_full.optimize(obj_full, n_trials=n_trials + 40)
    full_meta_p = st_full.best_params
    print(f"[{species.upper()}] Optuna Stage C Meta-Ensemble (5-Fold All) MAP@5={st_full.best_value:.5f} | mix_rank={full_meta_p['mix_rank']:.2f}, gamma={full_meta_p['gamma']:.2f}, delta={full_meta_p['delta']:+.3f}, a_o1={full_meta_p['a_o1']:.2f}, a_oc={full_meta_p['a_oc']:.2f}")
    return nested_preds_by_fold, full_meta_p, combine_fold


def main():
    t0 = time.time()
    os.makedirs("submissions_v2", exist_ok=True)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    folds = build_or_load_cv_v2()
    n_te = len(test_df)

    all_best_hp = {}
    all_meta_params = {}
    nested_cv_preds_final = {k: [None] * len(folds[k]["val_idx"]) for k in range(5)}
    single_family_cv_preds = {
        fam: {k: [None] * len(folds[k]["val_idx"]) for k in range(5)}
        for fam in ["LGBMRanker_LambdaRank", "XGBRanker_NDCG", "CatBoost_QuerySoftMax", "LGBM_Binary_Open", "XGB_Binary_Open", "CatBoost_Binary_Open"]
    }

    test_preds_v2 = [None] * n_te
    test_preds_v2_em = [None] * n_te

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        with open(os.path.join(CACHE_V2_DIR, f"v2_bundles_{sp}.pkl"), "rb") as f:
            data = pickle.load(f)

        outer_data = data["outer_data"]
        test_b = data["test_bundle"]
        full_seed_bundles = data["full_seed_bundles"]

        # 1) Optuna Stage B HPO on seed 2026 full sub-folds (zero test leakage!)
        hp = optuna_tune_stage2_hyperparams(sp, full_seed_bundles[2026], n_trials_tree=16)
        all_best_hp[sp] = hp

        # 2) Run leak-free 5-Fold Outer CV using Optuna-tuned models
        outer_preds_list = []
        for fd_data in outer_data:
            fk = fd_data["fold"]
            tf = time.time()
            cand_d, open_d = fit_all_stage2_models_on_subfolds(
                fd_data["sub_bundles"], fd_data["val_bundle"], sp, hp, seed=42 + 100 * fk
            )
            outer_preds_list.append({"fold": fk, "bundle": fd_data["val_bundle"], "cand": cand_d, "open": open_d})

            # Log individual family standalone CV predictions (for clear ablation table!)
            vb = fd_data["val_bundle"]
            vpos = vb["v_pos"]
            o_ens = (open_d["lgb_open"] + open_d["xgb_open"] + open_d["cat_open"]) / 3.0
            fam_pairs = [
                ("LGBMRanker_LambdaRank", cand_d["lgb_rank"], o_ens, 2.2),
                ("XGBRanker_NDCG", cand_d["xgb_rank"], o_ens, 2.2),
                ("CatBoost_QuerySoftMax", cand_d["cat_rank"], o_ens, 2.2),
                ("LGBM_Binary_Open", cand_d["lgb_bin"], open_d["lgb_open"], 1.0),
                ("XGB_Binary_Open", cand_d["xgb_bin"], open_d["xgb_open"], 1.0),
                ("CatBoost_Binary_Open", cand_d["cat_bin"], open_d["cat_open"], 1.0),
            ]
            for fname, cs_f, ol_f, gm_f in fam_pairs:
                pr_f = decode_top5(vb["cand_names"], cs_f, ol_f, gamma=gm_f, delta=0.0, raw_s1=vb["raw_s1"], max_sim_inl=vb["max_sim_inl"])
                for pi_loc, pr_i in zip(vpos, pr_f):
                    single_family_cv_preds[fname][fk][pi_loc] = pr_i
            print(f"  [{sp}] Outer Fold {fk} all 13 Stage-2 models trained in {time.time() - tf:.1f}s")

        # 3) Optuna Stage C Nested Meta-Ensemble & Graph-Cluster Post-Processing!
        nested_sp, full_meta_p, combine_fn = optuna_meta_blend_and_postproc(outer_preds_list, sp, n_trials=120)
        all_meta_params[sp] = full_meta_p
        for fk in range(5):
            vpos = outer_data[fk]["val_bundle"]["v_pos"]
            for pi_loc, pr_i in zip(vpos, nested_sp[fk]):
                nested_cv_preds_final[fk][pi_loc] = pr_i

        # 4) Full-Train Production Multi-Seed Ensemble on test.csv (2 seeds x 5 subfolds x 13 models = 130 models/species!)
        print(f"\n=== [{sp.upper()}] FULL-TRAIN MULTI-SEED (260-MODEL) TEST INFERENCE ===")
        cand_ens_t = None
        open_ens_t = None
        seeds = [2026, 7777]
        for sd in seeds:
            c_sd, o_sd = fit_all_stage2_models_on_subfolds(full_seed_bundles[sd], test_b, sp, hp, seed=sd)
            if cand_ens_t is None:
                cand_ens_t = {k: v / len(seeds) for k, v in c_sd.items()}
                open_ens_t = {k: v / len(seeds) for k, v in o_sd.items()}
            else:
                for k in c_sd:
                    cand_ens_t[k] += c_sd[k] / len(seeds)
                for k in o_sd:
                    open_ens_t[k] += o_sd[k] / len(seeds)

        cn_t, cp_t, ol_t = combine_fn({"bundle": test_b, "cand": cand_ens_t, "open": open_ens_t}, full_meta_p)

        # Check Saerens et al. EM implied test prior!
        probs_raw_t = expit(ol_t)
        pi_em = saerens_em_prior(probs_raw_t, PRIOR_PI_NEW[sp])
        em_shift = logit(np.clip(pi_em, 0.10, 0.85)) - logit(PRIOR_PI_NEW[sp])
        print(f"  [{sp.upper()}] Saerens EM estimated test pi_new = {pi_em:.4f} (KDE prior={PRIOR_PI_NEW[sp]:.3f}, EM logit shift={em_shift:+.3f}, Optuna outer delta={full_meta_p['delta']:+.3f})")

        pr_t_opt = decode_top5_from_probs(cn_t, cp_t, ol_t, delta=full_meta_p["delta"], raw_s1=test_b["raw_s1"], max_sim_inl=test_b["max_sim_inl"])
        # EM-blended shift version (half Optuna outer delta + half Saerens EM shift!)
        delta_em_blend = 0.5 * full_meta_p["delta"] + 0.5 * (full_meta_p["delta"] + 0.5 * em_shift)
        pr_t_em = decode_top5_from_probs(cn_t, cp_t, ol_t, delta=delta_em_blend, raw_s1=test_b["raw_s1"], max_sim_inl=test_b["max_sim_inl"])

        for loc_i, te_pos in enumerate(test_b["te_pos"]):
            test_preds_v2[te_pos] = " ".join(pr_t_opt[loc_i])
            test_preds_v2_em[te_pos] = " ".join(pr_t_em[loc_i])

        print(f"Completed full [{sp.upper()}] pipeline in {time.time() - ts:.1f}s")

    # Save all Optuna-discovered hyperparameters cleanly!
    with open(os.path.join(CACHE_V2_DIR, "optuna_stage2_best_params.json"), "w") as f:
        json.dump({"stage2_hp": all_best_hp, "stage3_meta": all_meta_params}, f, indent=2)

    print("\n================ V2 GROUND-UP 5-FOLD OPEN-SET CV SUMMARY ================")
    for fname in single_family_cv_preds:
        eval_cv_v2(folds, train_df, single_family_cv_preds[fname], verbose=True, tag=fname)
    res_opt = eval_cv_v2(folds, train_df, nested_cv_preds_final, verbose=True, tag="V2_OPTUNA_SUPER_ENSEMBLE")

    # Save verified V2 submissions + Super-Blend with verified 0.82895 v1!
    p_v2_main = "submissions_v2/submission_v2_optuna_champion.csv"
    p_v2_em = "submissions_v2/submission_v2_optuna_em_calibrated.csv"
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_v2}).to_csv(p_v2_main, index=False)
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_v2_em}).to_csv(p_v2_em, index=False)

    # Also create a 70% V2 Optuna Champion + 30% V1 (0.82895) Reciprocal-Rank Super-Blend!
    p_v2_v1 = "submissions_v2/submission_v2_plus_v1_blend.csv"
    if os.path.exists("submissions/submission_best.csv"):
        sub_v1 = pd.read_csv("submissions/submission_best.csv")
        rw = [1.0, 0.46, 0.28, 0.19, 0.14]
        bl_preds = []
        for i in range(n_te):
            sc = {}
            for r, tok in enumerate(test_preds_v2[i].split(" ")[:5]):
                sc[tok] = sc.get(tok, 0.0) + 0.70 * rw[r]
            for r, tok in enumerate(sub_v1["predictions"].iat[i].split(" ")[:5]):
                sc[tok] = sc.get(tok, 0.0) + 0.30 * rw[r]
            bl_preds.append(" ".join([k for k, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]]))
        pd.DataFrame({"id": test_df["id"].values, "predictions": bl_preds}).to_csv(p_v2_v1, index=False)
        verify_submission_file(p_v2_v1, test_df, train_df)

    # Copy V2 Optuna Champion to root submission.csv (keeping submissions/submission_best.csv untouched as 0.82895 backup!)
    pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_v2}).to_csv("submission.csv", index=False)
    verify_submission_file(p_v2_main, test_df, train_df)
    verify_submission_file(p_v2_em, test_df, train_df)
    verify_submission_file("submission.csv", test_df, train_df)

    for lbl, pth in [("V2_Optuna_Champion (submission.csv)", p_v2_main), ("V2_Optuna_EM_Calib", p_v2_em), ("V2_70_plus_V1_30_Blend", p_v2_v1)]:
        if not os.path.exists(pth):
            continue
        sdf = pd.read_csv(pth)
        r1 = sdf["predictions"].str.split(" ").str[0].values
        r2 = sdf["predictions"].str.split(" ").str[1].values
        for sp in ["hyena", "leopard"]:
            m = (test_df["species"].values == sp)
            print(f"[{lbl:36s} | {sp:7s}] Rank1==new: {(r1[m]=='new_individual').mean():.3f} | Rank2==new: {(r2[m]=='new_individual').mean():.3f}")

    print(f"\nALL V2 OPTUNA TUNING & TRAINING COMPLETE in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
