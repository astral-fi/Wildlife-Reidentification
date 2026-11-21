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

from cv_framework import build_or_load_cv, evaluate_cv_predictions, EST_PI_NEW, DATA_DIR
from pipeline_features import build_internal_subfolds, build_feature_bundle, CAND_FEATURE_NAMES

warnings.filterwarnings("ignore")


def make_cand_models(seed=42):
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.4, max_iter=350, random_state=seed)
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=1024, alpha=2.5, max_iter=85, early_stopping=True, random_state=seed)
        ),
        "lgb": lgb.LGBMClassifier(
            n_estimators=240, learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0, n_jobs=16, random_state=seed, verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=240, learning_rate=0.03, max_depth=4, min_child_weight=8,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
            n_jobs=16, random_state=seed, eval_metric="logloss", verbosity=0
        ),
        "cat": CatBoostClassifier(
            iterations=280, learning_rate=0.035, depth=5, l2_leaf_reg=7.0,
            thread_count=16, random_seed=seed, verbose=0
        ),
    }


def make_open_models(seed=42):
    return {
        "lgb": lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.03, num_leaves=12, max_depth=4,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=6.0, n_jobs=16, random_state=seed, verbose=-1
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=220, learning_rate=0.03, max_depth=3, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=6.0,
            n_jobs=16, random_state=seed, eval_metric="logloss", verbosity=0
        ),
        "cat": CatBoostClassifier(
            iterations=260, learning_rate=0.035, depth=4, l2_leaf_reg=8.0,
            thread_count=16, random_seed=seed, verbose=0
        ),
    }


def clip_logit(p):
    p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_c / (1.0 - p_c))


def stack_open_features(X_open_base, cand_logits_2d):
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
    known_mask = ~is_new
    new_mask = is_new
    w = np.where(is_new, pi_new / max(1, new_mask.sum()), (1.0 - pi_new) / max(1, known_mask.sum()))

    best_score = -1.0
    best_gd = (1.0, 0.0)
    for gamma in [0.65, 0.85, 1.05, 1.25, 1.50]:
        cond_probs = softmax(gamma * cand_logits, axis=1)[:, :5]
        hit_col = np.full(len(y_true), -1, dtype=np.int32)
        for r in range(5):
            m = (cand_names[:, r] == y_true)
            hit_col[m & (hit_col < 0)] = r
        valid_hit = (hit_col >= 0)
        vh_idx = np.where(valid_hit)[0]
        hc = hit_col[vh_idx]

        for delta in np.linspace(-0.90, 0.90, 19):
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
                best_gd = (gamma, float(delta))

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
    if not np.any(has_qq):
        return cand_names, cand_logits, open_logits

    open_out = open_logits.copy()
    nbr_open = (W_qq[has_qq] @ open_logits) / w_qq_sum[has_qq, 0]
    open_out[has_qq] = (1.0 - alpha_open) * open_logits[has_qq] + alpha_open * nbr_open

    probs = softmax(cand_logits, axis=1)
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


def train_species_grandmaster(
    gal_rows, target_bundles_list, species,
    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
    sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=4, seed=42, return_importances=False
):
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

    cand_keys = ["logreg", "mlp", "lgb", "xgb", "cat"]
    # Candidate ranking loves linear+MLP smoothness blended with GBDTs!
    cand_w = {"logreg": 0.16, "mlp": 0.12, "lgb": 0.24, "xgb": 0.24, "cat": 0.24}

    open_keys = ["lgb", "xgb", "cat"]
    # Open-set detection is strictly stronger with pure nonlinear GBDT trio!
    open_w = {"lgb": 0.32, "xgb": 0.34, "cat": 0.34}

    oof_cand_logits = {m: [] for m in cand_keys}
    target_cand_logits = [{m: np.zeros(tb["top_cands"].shape, dtype=np.float64) for m in cand_keys} for tb in target_bundles_list]
    cand_imp = np.zeros(len(CAND_FEATURE_NAMES), dtype=np.float64)

    for hold_s in range(n_subfolds):
        X_tr_list, y_tr_list = [], []
        for s_idx, sb in enumerate(sub_bundles):
            if s_idx == hold_s:
                continue
            nq, K, F = sb["X_cand"].shape
            y_match = (sb["cand_names"] == sb["y_true"][:, None]).astype(np.int32)
            keep_q = (~sb["is_new"]) | ((np.arange(nq) % 3) == 0)
            X_tr_list.append(sb["X_cand"][keep_q].reshape(-1, F))
            y_tr_list.append(y_match[keep_q].ravel())

        X_tr_c = np.vstack(X_tr_list)
        y_tr_c = np.concatenate(y_tr_list)
        hb = sub_bundles[hold_s]
        nq_h, K_h, F_h = hb["X_cand"].shape
        X_val_c = hb["X_cand"].reshape(-1, F_h)

        c_models = make_cand_models(seed + 17 * hold_s)
        for m in cand_keys:
            c_models[m].fit(X_tr_c, y_tr_c)
            if m == "lgb":
                cand_imp += c_models[m].feature_importances_ / float(n_subfolds)
            p_h = c_models[m].predict_proba(X_val_c)[:, 1]
            oof_cand_logits[m].append(clip_logit(p_h).reshape(nq_h, K_h))

            for t_i, tb in enumerate(target_bundles_list):
                nqt, Kt, Ft = tb["X_cand"].shape
                pt = c_models[m].predict_proba(tb["X_cand"].reshape(-1, Ft))[:, 1]
                target_cand_logits[t_i][m] += clip_logit(pt).reshape(nqt, Kt) / float(n_subfolds)

    oof_cand_blend = np.vstack([sum(cand_w[m] * oof_cand_logits[m][s] for m in cand_keys) for s in range(n_subfolds)])
    for s_idx, sb in enumerate(sub_bundles):
        bl_s = sum(cand_w[m] * oof_cand_logits[m][s_idx] for m in cand_keys)
        sb["X_open_stacked"] = stack_open_features(sb["X_open_base"], bl_s)

    for t_i, tb in enumerate(target_bundles_list):
        target_cand_logits[t_i]["blend"] = sum(cand_w[m] * target_cand_logits[t_i][m] for m in cand_keys)
        tb["X_open_stacked"] = stack_open_features(tb["X_open_base"], target_cand_logits[t_i]["blend"])

    oof_open_logits = {m: [] for m in open_keys}
    target_open_logits = [{m: np.zeros(len(tb["X_open_base"]), dtype=np.float64) for m in open_keys} for tb in target_bundles_list]
    open_imp = np.zeros(sub_bundles[0]["X_open_stacked"].shape[1], dtype=np.float64)

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
        for m in open_keys:
            o_models[m].fit(Xo_tr, yo_tr, sample_weight=wo_tr)
            if m == "lgb":
                open_imp += o_models[m].feature_importances_ / float(n_subfolds)
            po_h = clip_logit(o_models[m].predict_proba(hb["X_open_stacked"])[:, 1])
            oof_open_logits[m].append(po_h)
            for t_i, tb in enumerate(target_bundles_list):
                pot = clip_logit(o_models[m].predict_proba(tb["X_open_stacked"])[:, 1])
                target_open_logits[t_i][m] += pot / float(n_subfolds)

    oof_open_blend = np.concatenate([sum(open_w[m] * oof_open_logits[m][s] for m in open_keys) for s in range(n_subfolds)])
    for t_i in range(len(target_bundles_list)):
        target_open_logits[t_i]["blend"] = sum(open_w[m] * target_open_logits[t_i][m] for m in open_keys)

    oof_names = np.vstack([sb["cand_names"] for sb in sub_bundles])
    oof_y = np.concatenate([sb["y_true"] for sb in sub_bundles])
    oof_isn = np.concatenate([sb["is_new"] for sb in sub_bundles])

    # Tune (gamma, delta) strictly on inner OOF for both inductive-q and cluster-PP!
    (g_ind, d_ind), _ = tune_gamma_delta_map5(oof_names, oof_cand_blend, oof_open_blend, oof_y, oof_isn, pi_new)

    pp_names_l, pp_cand_l, pp_open_l = [], [], []
    offset = 0
    for sb in sub_bundles:
        n_s = len(sb["y_true"])
        cn_s, cl_s, ol_s = apply_query_cluster_postproc(
            sb["cand_names"], oof_cand_blend[offset:offset + n_s], oof_open_blend[offset:offset + n_s],
            sb["W_qq"], sb["w_qq_sum"], sb["has_qq"]
        )
        pp_names_l.append(cn_s)
        pp_cand_l.append(cl_s)
        pp_open_l.append(ol_s)
        offset += n_s
    (g_pp, d_pp), _ = tune_gamma_delta_map5(
        np.vstack(pp_names_l), np.vstack(pp_cand_l), np.concatenate(pp_open_l), oof_y, oof_isn, pi_new
    )

    out_targets = []
    for t_i, tb in enumerate(target_bundles_list):
        c_names = tb["cand_names"]
        c_log = target_cand_logits[t_i]["blend"]
        o_log = target_open_logits[t_i]["blend"]
        c_names_pp, c_log_pp, o_log_pp = apply_query_cluster_postproc(
            c_names, c_log, o_log, tb["W_qq"], tb["w_qq_sum"], tb["has_qq"]
        )
        out_targets.append({
            "G1_5Cand_GBDT3Open": {
                "preds": decode_from_scores(c_names, c_log, o_log, gamma=g_ind, delta=d_ind),
                "cand_names": c_names, "raw_cand_logits": c_log, "raw_open_logits": o_log,
                "gamma": g_ind, "delta": d_ind,
            },
            "G2_Grandmaster_ClusterPP": {
                "preds": decode_from_scores(c_names_pp, c_log_pp, o_log_pp, gamma=g_pp, delta=d_pp),
                "cand_names": c_names_pp, "raw_cand_logits": c_log_pp, "raw_open_logits": o_log_pp,
                "gamma": g_pp, "delta": d_pp,
            },
        })

    if return_importances:
        return out_targets, cand_imp, open_imp
    return out_targets


def verify_submission_file(sub_path, test_df, train_df):
    sub = pd.read_csv(sub_path)
    assert len(sub) == len(test_df) == 2791, f"Expected 2791 rows, got {len(sub)}"
    assert list(sub.columns) == ["id", "predictions"], f"Bad columns: {list(sub.columns)}"
    assert (sub["id"].values == test_df["id"].values).all(), "ID alignment mismatch!"
    assert sub["predictions"].isna().sum() == 0, "NaN predictions found!"

    valid_by_sp = {
        sp: set(train_df[train_df["species"] == sp]["identity"].unique()) | {"new_individual"}
        for sp in ["hyena", "leopard"]
    }
    for i, (pred_str, sp) in enumerate(zip(sub["predictions"].values, test_df["species"].values)):
        toks = pred_str.strip().split(" ")
        assert 1 <= len(toks) <= 5, f"Row {i} has {len(toks)} predictions!"
        assert len(set(toks)) == len(toks), f"Duplicate predictions in row {i}: {toks}"
        for t in toks:
            assert t in valid_by_sp[sp], f"Row {i} ({sp}) predicted invalid/wrong-species ID: {t}"
    print(f"[VERIFIED PASS] {sub_path}: 2,791 rows, 0 NaNs, exact Top-5 unique same-species IDs!")


def main():
    t0 = time.time()
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
    test_global_rows = np.arange(n_tr, n_tr + n_te, dtype=np.int32)

    fold_preds_g1 = {}
    fold_preds_g2 = {}
    outer_raw_store = {k: {} for k in range(5)}

    print("=== RUNNING EXP04 GRANDMASTER PIPELINE (LOIO-INVARIANT + WCCN + JACCARD + SIFT RANSAC) ===")
    for fold in cv["folds"]:
        fk = fold["fold"]
        tf = time.time()
        g_idx = fold["gallery_idx"]
        v_idx = fold["val_idx"]
        fold_preds_g1[fk] = [None] * len(v_idx)
        fold_preds_g2[fk] = [None] * len(v_idx)

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
            dec_val = train_species_grandmaster(
                g_rows, [val_bundle], sp,
                all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=4, seed=42 + 100 * fk
            )[0]

            for p_i, pr in zip(v_pos, dec_val["G1_5Cand_GBDT3Open"]["preds"]):
                fold_preds_g1[fk][p_i] = pr
            for p_i, pr in zip(v_pos, dec_val["G2_Grandmaster_ClusterPP"]["preds"]):
                fold_preds_g2[fk][p_i] = pr

            outer_raw_store[fk][sp] = {
                "v_pos": v_pos,
                "y_true": np.array(fold["val_labels"], dtype=object)[v_sp_m],
                "is_new": fold["val_is_new"][v_sp_m],
                "g2": dec_val["G2_Grandmaster_ClusterPP"],
                "g1": dec_val["G1_5Cand_GBDT3Open"],
            }
        print(f"Finished Fold {fk} in {time.time() - tf:.1f}s")

    # Also test G3: Leave-One-Outer-Fold-Out (gamma, delta) calibration across the 4 OTHER outer folds (zero leak!)
    fold_preds_g3 = {fk: [None] * len(cv["folds"][fk]["val_idx"]) for fk in range(5)}
    learned_outer_shifts = {"hyena": [], "leopard": []}
    for fk in range(5):
        for sp in ["hyena", "leopard"]:
            cn_tr = np.vstack([outer_raw_store[f][sp]["g2"]["cand_names"] for f in range(5) if f != fk])
            cl_tr = np.vstack([outer_raw_store[f][sp]["g2"]["raw_cand_logits"] for f in range(5) if f != fk])
            ol_tr = np.concatenate([outer_raw_store[f][sp]["g2"]["raw_open_logits"] for f in range(5) if f != fk])
            yt_tr = np.concatenate([outer_raw_store[f][sp]["y_true"] for f in range(5) if f != fk])
            in_tr = np.concatenate([outer_raw_store[f][sp]["is_new"] for f in range(5) if f != fk])

            (g_out, d_out), _ = tune_gamma_delta_map5(cn_tr, cl_tr, ol_tr, yt_tr, in_tr, EST_PI_NEW[sp])
            d_in = outer_raw_store[fk][sp]["g2"]["delta"]
            learned_outer_shifts[sp].append((g_out, d_out, d_out - d_in))

            cur = outer_raw_store[fk][sp]["g2"]
            preds_3 = decode_from_scores(cur["cand_names"], cur["raw_cand_logits"], cur["raw_open_logits"], gamma=g_out, delta=d_out)
            for p_i, pr in zip(outer_raw_store[fk][sp]["v_pos"], preds_3):
                fold_preds_g3[fk][p_i] = pr

    print("\n--- [G1_5Cand_GBDT3Open (LOIO-Invariant)] ---")
    r_g1 = evaluate_cv_predictions(cv, train_df, fold_preds_g1, verbose=True)
    print("\n--- [G2_Grandmaster_ClusterPP (Inner Calibrated)] ---")
    r_g2 = evaluate_cv_predictions(cv, train_df, fold_preds_g2, verbose=True)
    print("\n--- [G3_Grandmaster_OuterCrossCalibrated] ---")
    r_g3 = evaluate_cv_predictions(cv, train_df, fold_preds_g3, verbose=True)

    for sp in ["hyena", "leopard"]:
        arr = np.array(learned_outer_shifts[sp])
        print(f"[{sp}] Outer-vs-Inner calibration: mean gamma={arr[:,0].mean():.3f}, mean delta_outer={arr[:,1].mean():.3f}, mean shift (d_outer - d_inner)={arr[:,2].mean():+.3f}")

    # === FULL-TRAIN MULTI-SEED PRODUCTION ENSEMBLE FOR TEST.CSV ===
    print("\n=== TRAINING FULL-TRAIN MULTI-SEED GRANDMASTER ENSEMBLE FOR TEST.CSV ===")
    best_test_preds = [None] * n_te
    nopp_test_preds = [None] * n_te

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

        # 3 independent 5-subfold seeds (= 15 subfold splits x 8 models = 120 models per species!)
        seeds = [2026, 7777, 9999]
        decs = []
        cand_imp_tot = None
        for idx_s, sd in enumerate(seeds):
            if idx_s == 0:
                d_out, c_imp, _ = train_species_grandmaster(
                    full_g_rows, [test_bundle], sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                    sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=5, seed=sd, return_importances=True
                )
                cand_imp_tot = c_imp
            else:
                d_out = train_species_grandmaster(
                    full_g_rows, [test_bundle], sp,
                    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
                    sift_nbr_idx, sift_nbr_feats, X_norm_all, n_subfolds=5, seed=sd, return_importances=False
                )
            decs.append(d_out[0])

        top_imp_idx = np.argsort(-cand_imp_tot)[:12]
        print(f"[{sp.upper()}] Top-12 Candidate Reranker Features (LGBM):", [(CAND_FEATURE_NAMES[j], round(float(cand_imp_tot[j]), 1)) for j in top_imp_idx])

        # Apply outer-vs-inner shift only if G3 improved over G2 on CV!
        extra_shift = float(np.mean([x[2] for x in learned_outer_shifts[sp]])) if r_g3["lb_proxy_map5"] > r_g2["lb_proxy_map5"] + 1e-4 else 0.0

        for key_cfg, out_arr in [("G2_Grandmaster_ClusterPP", best_test_preds), ("G1_5Cand_GBDT3Open", nopp_test_preds)]:
            p_new_ens = np.mean([expit(d[key_cfg]["raw_open_logits"] + d[key_cfg]["delta"] + extra_shift) for d in decs], axis=0)
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

    sub_best_path = "submissions/submission_best.csv"
    sub_nopp_path = "submissions/submission_no_query_pp.csv"
    pd.DataFrame({"id": test_df["id"].values, "predictions": best_test_preds}).to_csv(sub_best_path, index=False)
    pd.DataFrame({"id": test_df["id"].values, "predictions": nopp_test_preds}).to_csv(sub_nopp_path, index=False)

    verify_submission_file(sub_best_path, test_df, train_df)
    verify_submission_file(sub_nopp_path, test_df, train_df)

    for label, pth in [("BEST_Grandmaster_ClusterPP", sub_best_path), ("Backup_NoQueryPP", sub_nopp_path)]:
        s_df = pd.read_csv(pth)
        r1 = s_df["predictions"].str.split(" ").str[0]
        r2 = s_df["predictions"].str.split(" ").str[1]
        for sp in ["hyena", "leopard"]:
            m = (test_df["species"].values == sp)
            print(
                f"[{label} | {sp}] Test Rank-1 `new_individual`={float((r1[m]=='new_individual').mean()):.3f} "
                f"(est prior={EST_PI_NEW[sp]:.3f}), Rank-2 `new_individual`={float((r2[m]=='new_individual').mean()):.3f}"
            )

    print(f"\nALL DONE in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
