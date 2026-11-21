import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.special import softmax, logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from cv_framework import build_or_load_cv, evaluate_cv_predictions, ap_at_5, EST_PI_NEW, TEST_SPECIES_WEIGHTS, DATA_DIR
from pipeline_features import build_internal_subfolds, build_feature_bundle

warnings.filterwarnings("ignore")


def make_cand_models(seed=42):
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.4, max_iter=400, random_state=seed)
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=2.5, max_iter=180, early_stopping=True, random_state=seed)
        ),
        "lgb": lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0, n_jobs=8, random_state=seed, verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=220, learning_rate=0.03, max_depth=4, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
            n_jobs=8, random_state=seed, eval_metric="logloss", verbosity=0
        ),
        "cat": CatBoostClassifier(
            iterations=280, learning_rate=0.035, depth=5, l2_leaf_reg=7.0,
            thread_count=8, random_seed=seed, verbose=0
        ),
    }


def make_open_models(seed=42):
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=400, random_state=seed)
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(48, 24), alpha=3.0, max_iter=200, early_stopping=True, random_state=seed)
        ),
        "lgb": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.03, num_leaves=12, max_depth=4,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=6.0, n_jobs=8, random_state=seed, verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=3, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
            n_jobs=8, random_state=seed, eval_metric="logloss", verbosity=0
        ),
        "cat": CatBoostClassifier(
            iterations=260, learning_rate=0.035, depth=4, l2_leaf_reg=8.0,
            thread_count=8, random_seed=seed, verbose=0
        ),
    }


def clip_logit(p):
    p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_c / (1.0 - p_c))


def stack_open_features(X_open_base, cand_logits_2d):
    """
    Append Stage-2 candidate reranker confidence summaries to X_open_base!
    cand_logits_2d: (n_q, K)
    """
    s_sorted = -np.sort(-cand_logits_2d, axis=1)
    l1 = s_sorted[:, 0]
    l2 = s_sorted[:, 1]
    probs = softmax(cand_logits_2d, axis=1)
    p1 = probs.max(axis=1)
    ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
    return np.column_stack([X_open_base, l1, l1 - l2, p1, ent]).astype(np.float32)


def decode_from_scores(cand_names, cand_logits, open_logits, gamma=1.0, delta=0.0):
    cond_probs = softmax(gamma * cand_logits, axis=1)
    p_new = expit(open_logits + delta)
    preds = []
    for i in range(len(cand_names)):
        kp = (1.0 - p_new[i]) * cond_probs[i, :5]
        items = [("new_individual", float(p_new[i]))] + [(cand_names[i, r], float(kp[r])) for r in range(5)]
        items.sort(key=lambda x: -x[1])
        preds.append([lbl for lbl, _ in items[:5]])
    return preds


def tune_gamma_delta_map5(cand_names, cand_logits, open_logits, y_true, is_new, pi_new):
    """
    Optimize (gamma, delta) strictly on inner gallery OOF to maximize exact prior-weighted MAP@5!
    """
    known_mask = ~is_new
    new_mask = is_new
    w = np.where(is_new, pi_new / max(1, new_mask.sum()), (1.0 - pi_new) / max(1, known_mask.sum()))

    best_score = -1.0
    best_gd = (1.0, 0.0)
    # Coarse-to-fine fast vectorized evaluation!
    for gamma in [0.65, 0.85, 1.05, 1.30, 1.60]:
        cond_probs = softmax(gamma * cand_logits, axis=1)[:, :5]  # (N, 5)
        # Precompute which rank 0..4 matches y_true for known rows (-1 if miss)
        hit_col = np.full(len(y_true), -1, dtype=np.int32)
        for r in range(5):
            m = (cand_names[:, r] == y_true)
            hit_col[m & (hit_col < 0)] = r

        for delta in np.linspace(-0.90, 0.90, 19):
            p_new = expit(open_logits + delta)
            kp = (1.0 - p_new[:, None]) * cond_probs  # (N, 5)
            # Rank of new_individual (1-based) = 1 + number of kp[:, r] > p_new
            rank_new = 1 + np.sum(kp > p_new[:, None], axis=1)
            ap_new = np.where(rank_new <= 5, 1.0 / rank_new, 0.0)

            # For known rows where hit_col >= 0: rank of hit_col is (hit_col + 1) + (1 if p_new >= kp[hit_col] else 0)
            ap_known = np.zeros(len(y_true), dtype=np.float64)
            valid_hit = (hit_col >= 0)
            vh_idx = np.where(valid_hit)[0]
            hc = hit_col[vh_idx]
            kp_hit = kp[vh_idx, hc]
            rk_hit = (hc + 1) + (p_new[vh_idx] >= kp_hit).astype(np.int32)
            ap_known[vh_idx] = np.where(rk_hit <= 5, 1.0 / rk_hit, 0.0)

            aps = np.where(is_new, ap_new, ap_known)
            sc = float(np.sum(aps * w))
            if sc > best_score:
                best_score = sc
                best_gd = (gamma, float(delta))

    # Fine-tune delta around best_gd
    g_opt, d_mid = best_gd
    cond_probs = softmax(g_opt * cand_logits, axis=1)[:, :5]
    hit_col = np.full(len(y_true), -1, dtype=np.int32)
    for r in range(5):
        m = (cand_names[:, r] == y_true)
        hit_col[m & (hit_col < 0)] = r
    valid_hit = (hit_col >= 0)
    vh_idx = np.where(valid_hit)[0]
    hc = hit_col[vh_idx]

    for delta in np.linspace(d_mid - 0.15, d_mid + 0.15, 31):
        p_new = expit(open_logits + delta)
        kp = (1.0 - p_new[:, None]) * cond_probs
        rank_new = 1 + np.sum(kp > p_new[:, None], axis=1)
        ap_new = np.where(rank_new <= 5, 1.0 / rank_new, 0.0)
        ap_known = np.zeros(len(y_true), dtype=np.float64)
        kp_hit = kp[vh_idx, hc]
        rk_hit = (hc + 1) + (p_new[vh_idx] >= kp_hit).astype(np.int32)
        ap_known[vh_idx] = np.where(rk_hit <= 5, 1.0 / rk_hit, 0.0)
        sc = float(np.sum(np.where(is_new, ap_new, ap_known) * w))
        if sc > best_score:
            best_score = sc
            best_gd = (g_opt, float(delta))

    return best_gd, best_score


def apply_query_cluster_postproc(cand_names, cand_logits, open_logits, W_qq, w_qq_sum, has_qq, alpha_open=0.30, alpha_cand=0.28):
    """
    Transductive query-cluster posterior consolidation across high-similarity query sightings!
    """
    if not np.any(has_qq):
        return cand_names, cand_logits, open_logits

    # 1. Smooth open-set log-odds across same-burst query neighbors
    open_out = open_logits.copy()
    nbr_open = (W_qq[has_qq] @ open_logits) / w_qq_sum[has_qq, 0]
    open_out[has_qq] = (1.0 - alpha_open) * open_logits[has_qq] + alpha_open * nbr_open

    # 2. Smooth candidate conditional probabilities across query neighbors (union of candidate IDs!)
    probs = softmax(cand_logits, axis=1)  # (N, K)
    cand_names_out = cand_names.copy()
    cand_logits_out = cand_logits.copy()

    for i in np.where(has_qq)[0]:
        nbrs = np.where(W_qq[i] > 0)[0]
        weights = W_qq[i, nbrs] / w_qq_sum[i, 0]
        score_map = {}
        for r in range(cand_names.shape[1]):
            cid = cand_names[i, r]
            score_map[cid] = score_map.get(cid, 0.0) + (1.0 - alpha_cand) * probs[i, r]
        for nj, wj in zip(nbrs, weights):
            for r in range(cand_names.shape[1]):
                cid = cand_names[nj, r]
                score_map[cid] = score_map.get(cid, 0.0) + alpha_cand * wj * probs[nj, r]
        sorted_items = sorted(score_map.items(), key=lambda x: -x[1])[:cand_names.shape[1]]
        for r, (cid, pr) in enumerate(sorted_items):
            cand_names_out[i, r] = cid
            cand_logits_out[i, r] = np.log(max(1e-8, pr))
    return cand_names_out, cand_logits_out, open_out


def train_species_stage2_and_predict(
    gal_rows, target_bundles_list, species,
    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
    sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=4, seed=42
):
    """
    Trains ALL 5 diverse model families strictly inside `gal_rows` via `n_subfolds` open-set sub-folds,
    computes inner OOF logits to stack & tune (gamma, delta), and predicts on every bundle in `target_bundles_list`!
    """
    pi_new = EST_PI_NEW[species]
    subfolds = build_internal_subfolds(
        gal_rows, all_identities[gal_rows], X_norm_all[gal_rows], n_splits=n_subfolds, seed=seed
    )

    sub_bundles = []
    for sf in subfolds:
        sg_glob = gal_rows[sf["sub_g_loc"]]
        sq_glob = gal_rows[sf["sub_q_loc"]]
        b = build_feature_bundle(
            sg_glob, sq_glob, species,
            all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
            sift_nbr_idx, sift_nbr_feats, top_k_cands=12
        )
        b["y_true"] = np.array(sf["sub_q_labels"], dtype=object)
        b["is_new"] = sf["sub_q_is_new"]
        sub_bundles.append(b)

    model_keys = ["logreg", "mlp", "lgb", "xgb", "cat"]
    blend_weights = {"logreg": 0.12, "mlp": 0.13, "lgb": 0.25, "xgb": 0.25, "cat": 0.25}

    # Step 1: Cross-fit Candidate Rerankers across the n_subfolds (train on n_subfolds-1, OOF on 1)
    # AND predict on target_bundles_list (averaging all n_subfolds models -> bagged ensemble!)
    oof_cand_logits = {m: [] for m in model_keys}
    target_cand_logits = [{m: np.zeros(tb["top_cands"].shape, dtype=np.float64) for m in model_keys} for tb in target_bundles_list]

    for hold_s in range(n_subfolds):
        # Assemble training candidate pairs from subfolds != hold_s (use known queries + 25% unseen queries as hard negatives!)
        X_tr_list, y_tr_list = [], []
        for s_idx, sb in enumerate(sub_bundles):
            if s_idx == hold_s:
                continue
            nq, K, F = sb["X_cand"].shape
            y_match = (sb["cand_names"] == sb["y_true"][:, None]).astype(np.int32)
            # Keep all known queries + every 3rd unseen query (so candidate logits also suppress impostors!)
            keep_q = (~sb["is_new"]) | ((np.arange(nq) % 3) == 0)
            X_tr_list.append(sb["X_cand"][keep_q].reshape(-1, F))
            y_tr_list.append(y_match[keep_q].ravel())

        X_tr_c = np.vstack(X_tr_list)
        y_tr_c = np.concatenate(y_tr_list)

        hb = sub_bundles[hold_s]
        nq_h, K_h, F_h = hb["X_cand"].shape
        X_val_c = hb["X_cand"].reshape(-1, F_h)

        c_models = make_cand_models(seed + 17 * hold_s)
        for m in model_keys:
            c_models[m].fit(X_tr_c, y_tr_c)
            p_h = c_models[m].predict_proba(X_val_c)[:, 1]
            oof_cand_logits[m].append(clip_logit(p_h).reshape(nq_h, K_h))

            for t_i, tb in enumerate(target_bundles_list):
                nqt, Kt, Ft = tb["X_cand"].shape
                pt = c_models[m].predict_proba(tb["X_cand"].reshape(-1, Ft))[:, 1]
                target_cand_logits[t_i][m] += clip_logit(pt).reshape(nqt, Kt) / float(n_subfolds)

    # Concatenate inner OOF candidate logits across subfolds
    oof_cand_cat = {m: np.vstack(oof_cand_logits[m]) for m in model_keys}
    oof_cand_cat["blend"] = sum(blend_weights[m] * oof_cand_cat[m] for m in model_keys)
    oof_cand_cat["stage1"] = np.vstack([sb["X_cand"][:, :, 0] for sb in sub_bundles]) * 18.0

    for t_i in range(len(target_bundles_list)):
        target_cand_logits[t_i]["blend"] = sum(blend_weights[m] * target_cand_logits[t_i][m] for m in model_keys)
        target_cand_logits[t_i]["stage1"] = target_bundles_list[t_i]["X_cand"][:, :, 0] * 18.0

    # Step 2: Build stacked Open-Set features (using blended OOF cand_logits!) & cross-fit Open-Set Detectors!
    for s_idx, sb in enumerate(sub_bundles):
        bl_s = sum(blend_weights[m] * oof_cand_logits[m][s_idx] for m in model_keys)
        sb["X_open_stacked"] = stack_open_features(sb["X_open_base"], bl_s)

    for t_i, tb in enumerate(target_bundles_list):
        tb["X_open_stacked"] = stack_open_features(tb["X_open_base"], target_cand_logits[t_i]["blend"])

    oof_open_logits = {m: [] for m in model_keys}
    target_open_logits = [{m: np.zeros(len(tb["X_open_base"]), dtype=np.float64) for m in model_keys} for tb in target_bundles_list]

    for hold_s in range(n_subfolds):
        Xo_list, yo_list, wo_list = [], [], []
        for s_idx, sb in enumerate(sub_bundles):
            if s_idx == hold_s:
                continue
            isn = sb["is_new"]
            Xo_list.append(sb["X_open_stacked"])
            yo_list.append(isn.astype(np.int32))
            n_n = max(1, int(isn.sum()))
            n_k = max(1, int((~isn).sum()))
            w_s = np.where(isn, pi_new / n_n, (1.0 - pi_new) / n_k) * len(isn)
            wo_list.append(w_s)

        Xo_tr = np.vstack(Xo_list)
        yo_tr = np.concatenate(yo_list)
        wo_tr = np.concatenate(wo_list)

        hb = sub_bundles[hold_s]
        o_models = make_open_models(seed + 31 * hold_s)
        for m in model_keys:
            if m == "mlp":
                # MLPClassifier in sklearn<1.9 may not accept sample_weight when early_stopping=True; resample/prior-shift!
                o_models[m].fit(Xo_tr, yo_tr)
                raw_emp_pi = yo_tr.mean()
                prior_shift = logit(pi_new) - logit(raw_emp_pi)
                po_h = clip_logit(o_models[m].predict_proba(hb["X_open_stacked"])[:, 1]) + prior_shift
                oof_open_logits[m].append(po_h)
                for t_i, tb in enumerate(target_bundles_list):
                    pot = clip_logit(o_models[m].predict_proba(tb["X_open_stacked"])[:, 1]) + prior_shift
                    target_open_logits[t_i][m] += pot / float(n_subfolds)
            else:
                if m == "logreg":
                    o_models[m].fit(Xo_tr, yo_tr, logisticregression__sample_weight=wo_tr)
                else:
                    o_models[m].fit(Xo_tr, yo_tr, sample_weight=wo_tr)
                po_h = clip_logit(o_models[m].predict_proba(hb["X_open_stacked"])[:, 1])
                oof_open_logits[m].append(po_h)
                for t_i, tb in enumerate(target_bundles_list):
                    pot = clip_logit(o_models[m].predict_proba(tb["X_open_stacked"])[:, 1])
                    target_open_logits[t_i][m] += pot / float(n_subfolds)

    oof_open_cat = {m: np.concatenate(oof_open_logits[m]) for m in model_keys}
    oof_open_cat["blend"] = sum(blend_weights[m] * oof_open_cat[m] for m in model_keys)

    for t_i in range(len(target_bundles_list)):
        target_open_logits[t_i]["blend"] = sum(blend_weights[m] * target_open_logits[t_i][m] for m in model_keys)

    # Concatenate OOF labels & candidate names to tune (gamma, delta) strictly on inner OOF!
    oof_names = np.vstack([sb["cand_names"] for sb in sub_bundles])
    oof_y = np.concatenate([sb["y_true"] for sb in sub_bundles])
    oof_isn = np.concatenate([sb["is_new"] for sb in sub_bundles])

    configs_map = {
        "M0_Stage1_Retrieval_LogRegOpen": ("stage1", "logreg", False),
        "M1_LogisticRegression_Stack": ("logreg", "logreg", False),
        "M2_MLP_Neural_FromScratch": ("mlp", "mlp", False),
        "M3_LightGBM_Rerank_Open": ("lgb", "lgb", False),
        "M4_CatBoost_Rerank_Open": ("cat", "cat", False),
        "M5_XGBoost_Rerank_Open": ("xgb", "xgb", False),
        "M6_Full_5Model_Ensemble": ("blend", "blend", False),
        "M7_Full_Ensemble_ClusterPostProc": ("blend", "blend", True),
    }

    tuned_params = {}
    for cfg_name, (ck, ok, use_pp) in configs_map.items():
        if not use_pp:
            (g_opt, d_opt), _ = tune_gamma_delta_map5(oof_names, oof_cand_cat[ck], oof_open_cat[ok], oof_y, oof_isn, pi_new)
        else:
            # Apply query-cluster postproc per subfold on inner OOF too before tuning!
            pp_names_l, pp_cand_l, pp_open_l = [], [], []
            offset = 0
            for s_idx, sb in enumerate(sub_bundles):
                n_s = len(sb["y_true"])
                cn_s, cl_s, ol_s = apply_query_cluster_postproc(
                    sb["cand_names"],
                    oof_cand_cat[ck][offset:offset + n_s],
                    oof_open_cat[ok][offset:offset + n_s],
                    sb["W_qq"], sb["w_qq_sum"], sb["has_qq"]
                )
                pp_names_l.append(cn_s)
                pp_cand_l.append(cl_s)
                pp_open_l.append(ol_s)
                offset += n_s
            (g_opt, d_opt), _ = tune_gamma_delta_map5(
                np.vstack(pp_names_l), np.vstack(pp_cand_l), np.concatenate(pp_open_l), oof_y, oof_isn, pi_new
            )
        tuned_params[cfg_name] = (g_opt, d_opt)

    # Decode predictions for every target bundle!
    target_decoded = []
    for t_i, tb in enumerate(target_bundles_list):
        dec_dict = {}
        for cfg_name, (ck, ok, use_pp) in configs_map.items():
            g_opt, d_opt = tuned_params[cfg_name]
            c_names = tb["cand_names"]
            c_log = target_cand_logits[t_i][ck]
            o_log = target_open_logits[t_i][ok]
            if use_pp:
                c_names, c_log, o_log = apply_query_cluster_postproc(
                    c_names, c_log, o_log, tb["W_qq"], tb["w_qq_sum"], tb["has_qq"]
                )
            dec_dict[cfg_name] = {
                "preds": decode_from_scores(c_names, c_log, o_log, gamma=g_opt, delta=d_opt),
                "cand_names": c_names,
                "cand_logits": c_log * g_opt,
                "open_logits": o_log + d_opt,
            }
        target_decoded.append(dec_dict)

    return target_decoded


def main():
    t_start = time.time()
    os.makedirs("submissions", exist_ok=True)

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

    cv = build_or_load_cv()
    cfg_names = [
        "M0_Stage1_Retrieval_LogRegOpen",
        "M1_LogisticRegression_Stack",
        "M2_MLP_Neural_FromScratch",
        "M3_LightGBM_Rerank_Open",
        "M4_CatBoost_Rerank_Open",
        "M5_XGBoost_Rerank_Open",
        "M6_Full_5Model_Ensemble",
        "M7_Full_Ensemble_ClusterPostProc",
    ]
    fold_preds_all = {c: {} for c in cfg_names}

    # Also collect outer-CV bagged test predictions across the 5 outer folds as a bonus ensemble component!
    test_global_rows = np.arange(n_tr, n_tr + n_te, dtype=np.int32)

    print("=== RUNNING 5-FOLD OPEN-SET CV FOR STAGE-2 STACKING ENSEMBLE ===")
    top12_recalls = {"hyena": [], "leopard": []}

    for fold in cv["folds"]:
        fk = fold["fold"]
        tf = time.time()
        g_idx = fold["gallery_idx"]
        v_idx = fold["val_idx"]
        v_lbls = np.array(fold["val_labels"], dtype=object)
        v_new = fold["val_is_new"]

        for c in cfg_names:
            fold_preds_all[c][fk] = [None] * len(v_idx)

        for sp in ["hyena", "leopard"]:
            g_sp_m = (train_df["species"].values[g_idx] == sp)
            v_sp_m = (train_df["species"].values[v_idx] == sp)
            g_rows = g_idx[g_sp_m]
            v_rows = v_idx[v_sp_m]
            v_pos = np.where(v_sp_m)[0]

            val_bundle = build_feature_bundle(
                g_rows, v_rows, sp,
                all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                sift_nbr_idx, sift_nbr_feats, top_k_cands=12
            )
            # Check Top-12 candidate recall on known queries!
            known_local = ~v_new[v_sp_m]
            y_known_sp = v_lbls[v_sp_m][known_local]
            rec12 = np.mean([(y_known_sp[i] in val_bundle["cand_names"][known_local][i]) for i in range(len(y_known_sp))])
            top12_recalls[sp].append(rec12)

            dec_list = train_species_stage2_and_predict(
                g_rows, [val_bundle], sp,
                all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=4, seed=42 + 100 * fk
            )
            val_dec = dec_list[0]
            for c in cfg_names:
                for p_i, pr in zip(v_pos, val_dec[c]["preds"]):
                    fold_preds_all[c][fk][p_i] = pr

        print(f"Completed Outer Fold {fk} in {time.time() - tf:.1f}s | Top-12 Known Recall: Hyena={top12_recalls['hyena'][-1]:.4f}, Leopard={top12_recalls['leopard'][-1]:.4f}")

    print(f"\nMean Top-12 Candidate Recall on Known Sightings: Hyena={np.mean(top12_recalls['hyena']):.4f}, Leopard={np.mean(top12_recalls['leopard']):.4f}")

    summary_rows = []
    for c in cfg_names:
        print(f"\n--- [{c}] ---")
        res = evaluate_cv_predictions(cv, train_df, fold_preds_all[c], verbose=True)
        res["name"] = c
        summary_rows.append(res)

    df_sum = pd.DataFrame(summary_rows)[["name", "lb_proxy_map5", "hyena_proxy_map5", "leopard_proxy_map5", "hyena_known_map5", "hyena_new_map5", "leopard_known_map5", "leopard_new_map5"]]
    print("\n================ EXP03 STAGE-2 STACKING SUMMARY ================")
    print(df_sum.sort_values("lb_proxy_map5", ascending=False).to_string(index=False))
    df_sum.to_csv("cache/exp03_results.csv", index=False)

    # === TRAIN FULL PRODUCTION MULTI-SEED ENSEMBLE ON 100% OF TRAIN.CSV FOR TEST.CSV ===
    print("\n=== TRAINING FULL-TRAIN PRODUCTION ENSEMBLE (2 SEEDS x 5 SUBFOLDS x 5 FAMILIES) FOR TEST.CSV ===")
    t_sub = time.time()
    test_preds_m6 = [None] * n_te
    test_preds_m7 = [None] * n_te

    for sp in ["hyena", "leopard"]:
        full_g_rows = np.where(train_df["species"].values == sp)[0].astype(np.int32)
        te_sp_mask = (test_df["species"].values == sp)
        te_pos = np.where(te_sp_mask)[0]
        te_q_rows = test_global_rows[te_sp_mask]

        test_bundle = build_feature_bundle(
            full_g_rows, te_q_rows, sp,
            all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
            sift_nbr_idx, sift_nbr_feats, top_k_cands=12
        )

        # Multi-seed bagging across 2 independent 5-subfold splits (= 10 subfolds x 5 model families = 50 models!)
        dec_seed1 = train_species_stage2_and_predict(
            full_g_rows, [test_bundle], sp,
            all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
            sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=5, seed=2026
        )[0]
        dec_seed2 = train_species_stage2_and_predict(
            full_g_rows, [test_bundle], sp,
            all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
            sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=5, seed=7777
        )[0]

        for cfg_key, target_arr in [("M6_Full_5Model_Ensemble", test_preds_m6), ("M7_Full_Ensemble_ClusterPostProc", test_preds_m7)]:
            d1, d2 = dec_seed1[cfg_key], dec_seed2[cfg_key]
            # Average probabilities across both 5-subfold seeds!
            p_new_avg = 0.5 * expit(d1["open_logits"]) + 0.5 * expit(d2["open_logits"])
            cp1 = softmax(d1["cand_logits"], axis=1)
            cp2 = softmax(d2["cand_logits"], axis=1)
            for idx_local, pos_te in enumerate(te_pos):
                p_n = float(p_new_avg[idx_local])
                cand_score = {"new_individual": p_n}
                for r in range(test_bundle["cand_names"].shape[1]):
                    c1 = d1["cand_names"][idx_local, r]
                    cand_score[c1] = cand_score.get(c1, 0.0) + 0.5 * (1.0 - p_n) * float(cp1[idx_local, r])
                    c2 = d2["cand_names"][idx_local, r]
                    cand_score[c2] = cand_score.get(c2, 0.0) + 0.5 * (1.0 - p_n) * float(cp2[idx_local, r])
                top5 = [lbl for lbl, _ in sorted(cand_score.items(), key=lambda x: -x[1])[:5]]
                target_arr[pos_te] = " ".join(top5)

    sub_m7 = pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_m7})
    sub_m6 = pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds_m6})
    sub_m7.to_csv("submissions/submission_best_full_ensemble_pp.csv", index=False)
    sub_m6.to_csv("submissions/submission_full_ensemble_inductive_q.csv", index=False)

    # Print diagnostics on test predictions!
    for label, s_df in [("M7_Best_ClusterPP", sub_m7), ("M6_Ensemble", sub_m6)]:
        top1_lbls = s_df["predictions"].str.split(" ").str[0]
        has_new_in5 = s_df["predictions"].str.contains("new_individual")
        for sp in ["hyena", "leopard"]:
            m = (test_df["species"].values == sp)
            r1_new = (top1_lbls[m] == "new_individual").mean()
            r5_new = has_new_in5[m].mean()
            print(f"[{label} - {sp}] Test Rank-1 `new_individual` rate: {r1_new:.3f} (est prior={EST_PI_NEW[sp]:.3f}), `new_individual` in Top-5: {r5_new:.3f}")

    print(f"Saved submissions in {time.time() - t_sub:.1f}s (Total script runtime: {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
