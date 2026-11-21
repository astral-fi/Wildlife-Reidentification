import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.sparse.csgraph import connected_components
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from cv_framework import build_or_load_cv, evaluate_cv_predictions, EST_PI_NEW, DATA_DIR


def precompute_pca_basis(Xg, Xq, signed_pow=1.0):
    """Compute exact centered eigenbasis ONCE per (fold, species) in ~0.3s!"""
    if signed_pow != 1.0:
        Xg = np.sign(Xg) * (np.abs(Xg) ** signed_pow)
        Xq = np.sign(Xq) * (np.abs(Xq) ** signed_pow)
    Xg_n = normalize(Xg, axis=1).astype(np.float64)
    Xq_n = normalize(Xq, axis=1).astype(np.float64)
    X_all = np.vstack([Xg_n, Xq_n])
    Xc = X_all - X_all.mean(axis=0, keepdims=True)
    N, D = Xc.shape
    if N < D:
        G = (Xc @ Xc.T) / N
        eigvals, U = np.linalg.eigh(G)
        m = eigvals > 1e-8
        eigvals = eigvals[m]
        proj = U[:, m] * np.sqrt(N * eigvals)[None, :]  # equals Xc @ V!
    else:
        C = (Xc.T @ Xc) / N
        eigvals, V = np.linalg.eigh(C)
        m = eigvals > 1e-8
        eigvals = eigvals[m]
        proj = Xc @ V[:, m]
    return proj, eigvals, len(Xg)


def apply_whiten_from_basis(proj, eigvals, n_g, power=0.65, reg=2e-3, top_d=None):
    if top_d is not None and proj.shape[1] > top_d:
        proj = proj[:, -top_d:]
        eigvals = eigvals[-top_d:]
    Z = proj * ((eigvals + reg) ** (-0.5 * power))[None, :]
    Z = normalize(Z, axis=1).astype(np.float32)
    return Z[:n_g], Z[n_g:]


def fast_shrinkage_lda(Zg, yg, Zq, shrinkage=0.20, n_comp=220, max_in_dim=512):
    if Zg.shape[1] > max_in_dim:
        Zg = normalize(Zg[:, -max_in_dim:], axis=1)
        Zq = normalize(Zq[:, -max_in_dim:], axis=1)
    vc = pd.Series(yg).value_counts()
    valid = set(vc[vc >= 2].index)
    mask = np.array([y in valid for y in yg])
    n_cls = len(valid)
    d_out = min(n_comp, n_cls - 1, Zg.shape[1])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage, n_components=d_out)
    lda.fit(Zg[mask], yg[mask])
    Lg = normalize(lda.transform(Zg), axis=1).astype(np.float32)
    Lq = normalize(lda.transform(Zq), axis=1).astype(np.float32)
    return Lg, Lq


def compute_burst_ids(X_norm, identities, thresh=0.875):
    burst_id = np.full(len(identities), -1, dtype=np.int32)
    nxt = 0
    for ident in np.unique(identities):
        idx = np.where(identities == ident)[0]
        if len(idx) == 1:
            burst_id[idx[0]] = nxt
            nxt += 1
        else:
            adj = (X_norm[idx] @ X_norm[idx].T) >= thresh
            nc, lbl = connected_components(adj, directed=False)
            for c in range(nc):
                burst_id[idx[lbl == c]] = nxt
                nxt += 1
    return burst_id


def aggregate_class_matrices(S, gal_ids, unique_classes=None, class_inv=None):
    if unique_classes is None:
        unique_classes, class_inv = np.unique(gal_ids, return_inverse=True)
    n_q = S.shape[0]
    n_c = len(unique_classes)
    C_max = np.full((n_q, n_c), -1e9, dtype=np.float64)
    C_lse = np.full((n_q, n_c), -1e9, dtype=np.float64)
    tau = 16.0
    for c in range(n_c):
        cols = np.where(class_inv == c)[0]
        sub = S[:, cols]
        mx = sub.max(axis=1)
        C_max[:, c] = mx
        if sub.shape[1] == 1:
            C_lse[:, c] = mx
        else:
            valid = sub > -1e5
            exps = np.where(valid, np.exp(np.clip(tau * (sub - mx[:, None]), -50, 0)), 0.0)
            C_lse[:, c] = mx + np.log(np.maximum(1e-12, exps.sum(axis=1))) / tau
    return C_max, C_lse, unique_classes, class_inv


def identity_level_csls(C_mat, k_imp=15):
    s_part = -np.sort(-C_mat, axis=1)[:, 1:min(k_imp + 1, C_mat.shape[1])]
    bg_mean = s_part.mean(axis=1, keepdims=True)
    return C_mat - 0.5 * bg_mean


def decoupled_calibrate_and_predict(
    C_rank_gg_lobo, C_open_gg_lobo, true_cls_lobo,
    C_open_gg_loio,
    C_rank_val, C_open_val,
    unique_classes, pi_new, extra_open_lobo=None, extra_open_loio=None, extra_open_val=None
):
    topM = min(15, len(unique_classes))
    top_idx_lobo = np.argsort(-C_rank_gg_lobo, axis=1)[:, :topM]
    top_sc_lobo = np.take_along_axis(C_rank_gg_lobo, top_idx_lobo, axis=1)
    hit_mask = (top_idx_lobo == true_cls_lobo[:, None])
    has_hit = hit_mask.any(axis=1)
    hit_pos = hit_mask[has_hit].argmax(axis=1)
    top_sc_hit = top_sc_lobo[has_hit]

    def nll_tau(tau):
        logits = tau * (top_sc_hit - top_sc_hit[:, :1])
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        return -np.mean(log_probs[np.arange(len(hit_pos)), hit_pos])

    tau_opt = float(minimize_scalar(nll_tau, bounds=(2.0, 80.0), method="bounded").x)

    def open_feats(C_op, extra=None):
        s_op = -np.sort(-C_op, axis=1)[:, :10]
        s1, s2, s3, s5 = s_op[:, 0], s_op[:, 1], s_op[:, 2], s_op[:, 4]
        bg = s_op[:, 1:10].mean(axis=1)
        bg_std = s_op[:, 1:10].std(axis=1) + 1e-4
        z1 = (s1 - bg) / bg_std
        cols = [s1, s2, s1 - s2, s1 - s3, s1 - s5, bg, z1]
        if extra is not None:
            cols.append(extra)
        return np.column_stack(cols)

    X_known = open_feats(C_open_gg_lobo, extra=extra_open_lobo)
    X_new = open_feats(C_open_gg_loio, extra=extra_open_loio)
    X_cal = np.vstack([X_known, X_new])
    y_cal = np.array([0] * len(X_known) + [1] * len(X_new))
    sw = np.array([(1.0 - pi_new) / len(X_known)] * len(X_known) + [pi_new / len(X_new)] * len(X_new)) * len(X_cal)

    clf = LogisticRegression(C=2.0, max_iter=500)
    clf.fit(X_cal, y_cal, sample_weight=sw)

    p_new = clf.predict_proba(open_feats(C_open_val, extra=extra_open_val))[:, 1]
    top_idx_val = np.argsort(-C_rank_val, axis=1)[:, :topM]
    top_sc_val = np.take_along_axis(C_rank_val, top_idx_val, axis=1)
    cond_probs = softmax(tau_opt * top_sc_val, axis=1)

    preds = []
    for i in range(len(C_rank_val)):
        cands = [unique_classes[j] for j in top_idx_val[i, :5]]
        known_post = (1.0 - p_new[i]) * cond_probs[i, :5]
        items = [("new_individual", p_new[i])] + [(cands[r], known_post[r]) for r in range(5)]
        items.sort(key=lambda x: -x[1])
        preds.append([lbl for lbl, _ in items[:5]])
    return preds


def lookup_sift_bonus_from_cache(q_global_rows, g_global_rows, gal_class_inv, n_classes, allowed_mask, sift_nbr_idx, sift_nbr_feats):
    """
    O(1) lookup from precomputed global top-36 SIFT+RANSAC pair cache! Runs in ~0.01s per fold!
    allowed_mask: (n_q, n_g) boolean mask of valid gallery entries (excludes same-burst in LOBO or same-ID in LOIO).
    """
    g_to_local = np.full(len(sift_nbr_idx), -1, dtype=np.int32)
    g_to_local[g_global_rows] = np.arange(len(g_global_rows), dtype=np.int32)

    n_q = len(q_global_rows)
    C_bonus = np.zeros((n_q, n_classes), dtype=np.float64)
    max_inl = np.zeros(n_q, dtype=np.float32)
    max_r72 = np.zeros(n_q, dtype=np.float32)

    nbrs = sift_nbr_idx[q_global_rows]       # (n_q, K)
    feats = sift_nbr_feats[q_global_rows]    # (n_q, K, 6): [r80, r72, mut, margin, inliers, inl_ratio]

    for qi in range(n_q):
        loc_js = g_to_local[nbrs[qi]]
        for m in range(nbrs.shape[1]):
            lj = loc_js[m]
            if lj < 0 or not allowed_mask[qi, lj]:
                continue
            r72, mut, inl = feats[qi, m, 1], feats[qi, m, 2], feats[qi, m, 4]
            if inl > max_inl[qi]:
                max_inl[qi] = inl
            if r72 > max_r72[qi]:
                max_r72[qi] = r72
            b = 0.012 * np.log1p(inl) + 0.006 * np.log1p(r72) + 0.004 * np.log1p(mut) + 0.025 * (inl >= 8)
            cls_c = gal_class_inv[lj]
            if b > C_bonus[qi, cls_c]:
                C_bonus[qi, cls_c] = b

    extra_open = np.column_stack([np.log1p(max_inl), np.log1p(max_r72), (max_inl >= 7).astype(np.float32)])
    return C_bonus, extra_open


def run_suite():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    miew_all = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))[:len(train_df)]
    X_norm_base = normalize(miew_all, axis=1)
    cv = build_or_load_cv()
    vlad_tr = np.load("cache/vlad_c64_p512.npy")[:len(train_df)]

    has_sift_cache = os.path.exists("cache/sift_top36_global.npz")
    if has_sift_cache:
        sift_z = np.load("cache/sift_top36_global.npz")
        sift_nbr_idx = sift_z["nbr_idx"]
        sift_nbr_feats = sift_z["nbr_feats"]

    configs = [
        ("A1_DualWhiten_p0.50", {}),
        ("A2_DualWhiten_p0.65", {}),
        ("A3_DualWhiten_p0.75_reg5e3", {}),
        ("A4_SignedPow0.85_Whiten_p0.65", {}),
        ("B1_Decoupled_WhitenOpen_LSE_Rank", {}),
        ("B2_Decoupled_WhitenOpen_LSE_IdCSLS", {}),
        ("B3_Decoupled_Plus_ShrinkageLDA", {}),
        ("B4_Decoupled_LDA_Plus_RidgeClf", {}),
        ("B5_Decoupled_LDA_Ridge_Plus_VLAD", {}),
        ("B6_Decoupled_All_Plus_QueryClusterQE", {}),
    ]
    if has_sift_cache:
        configs.append(("B7_Decoupled_All_Plus_SIFT_RANSAC", {}))

    fold_preds_by_cfg = {name: {} for name, _ in configs}

    for fold in cv["folds"]:
        fk = fold["fold"]
        tf = time.time()
        g_idx = fold["gallery_idx"]
        v_idx = fold["val_idx"]
        for name, _ in configs:
            fold_preds_by_cfg[name][fk] = [None] * len(v_idx)

        for sp in ["hyena", "leopard"]:
            g_sp_mask = (train_df["species"].values[g_idx] == sp)
            v_sp_mask = (train_df["species"].values[v_idx] == sp)
            g_rows = g_idx[g_sp_mask]
            v_rows = v_idx[v_sp_mask]
            v_pos = np.where(v_sp_mask)[0]

            Xg_raw, Xq_raw = miew_all[g_rows], miew_all[v_rows]
            yg = train_df["identity"].values[g_rows]
            gal_bursts = compute_burst_ids(X_norm_base[g_rows], yg, thresh=0.875)
            u_cls, c_inv = np.unique(yg, return_inverse=True)

            same_id = (yg[:, None] == yg[None, :])
            same_burst = (gal_bursts[:, None] == gal_bursts[None, :])
            lobo_rows = np.where((same_id & (~same_burst)).sum(axis=1) > 0)[0]
            true_cls_lobo = c_inv[lobo_rows]

            # Precompute eigenbases ONCE per (fold, species)!
            proj_10, eig_10, n_g = precompute_pca_basis(Xg_raw, Xq_raw, signed_pow=1.0)
            proj_85, eig_85, _ = precompute_pca_basis(Xg_raw, Xq_raw, signed_pow=0.85)

            for cname, p, reg, pr_use, eg_use in [
                ("A1_DualWhiten_p0.50", 0.50, 1e-3, proj_10, eig_10),
                ("A2_DualWhiten_p0.65", 0.65, 2e-3, proj_10, eig_10),
                ("A3_DualWhiten_p0.75_reg5e3", 0.75, 5e-3, proj_10, eig_10),
                ("A4_SignedPow0.85_Whiten_p0.65", 0.65, 2e-3, proj_85, eig_85),
            ]:
                Zg, Zq = apply_whiten_from_basis(pr_use, eg_use, n_g, power=p, reg=reg)
                S_gg = Zg @ Zg.T
                S_qg = Zq @ Zg.T
                S_lobo = S_gg[lobo_rows].copy()
                S_lobo[same_burst[lobo_rows]] = -1e9
                S_loio = S_gg.copy()
                S_loio[same_id] = -1e9

                Cm_lobo, _, _, _ = aggregate_class_matrices(S_lobo, yg, u_cls, c_inv)
                Cm_loio, _, _, _ = aggregate_class_matrices(S_loio, yg, u_cls, c_inv)
                Cm_val, _, _, _ = aggregate_class_matrices(S_qg, yg, u_cls, c_inv)

                preds = decoupled_calibrate_and_predict(
                    Cm_lobo, Cm_lobo, true_cls_lobo, Cm_loio, Cm_val, Cm_val, u_cls, EST_PI_NEW[sp]
                )
                for p_i, pr in zip(v_pos, preds):
                    fold_preds_by_cfg[cname][fk][p_i] = pr

            # Multi-subspace features for B1..B7
            Zg_w, Zq_w = apply_whiten_from_basis(proj_85, eig_85, n_g, power=0.65, reg=2e-3)
            Lg, Lq = fast_shrinkage_lda(Zg_w, yg, Zq_w, shrinkage=0.20, n_comp=220)
            proj_vl, eig_vl, _ = precompute_pca_basis(vlad_tr[g_rows], vlad_tr[v_rows], signed_pow=1.0)
            Vg_vlad, Vq_vlad = apply_whiten_from_basis(proj_vl, eig_vl, n_g, power=0.40, reg=1e-2)

            S_w_gg = Zg_w @ Zg_w.T
            S_w_qg = Zq_w @ Zg_w.T
            S_lda_gg = Lg @ Lg.T
            S_lda_qg = Lq @ Lg.T
            S_vlad_gg = Vg_vlad @ Vg_vlad.T
            S_vlad_qg = Vq_vlad @ Vg_vlad.T

            Sw_lobo = S_w_gg[lobo_rows].copy()
            Sw_lobo[same_burst[lobo_rows]] = -1e9
            Sw_loio = S_w_gg.copy()
            Sw_loio[same_id] = -1e9
            C_open_lobo, C_lse_w_lobo, _, _ = aggregate_class_matrices(Sw_lobo, yg, u_cls, c_inv)
            C_open_loio, _, _, _ = aggregate_class_matrices(Sw_loio, yg, u_cls, c_inv)
            C_open_val, C_lse_w_val, _, _ = aggregate_class_matrices(S_w_qg, yg, u_cls, c_inv)

            Slda_lobo = S_lda_gg[lobo_rows].copy()
            Slda_lobo[same_burst[lobo_rows]] = -1e9
            _, C_lse_lda_lobo, _, _ = aggregate_class_matrices(Slda_lobo, yg, u_cls, c_inv)
            _, C_lse_lda_val, _, _ = aggregate_class_matrices(S_lda_qg, yg, u_cls, c_inv)

            Svlad_lobo = S_vlad_gg[lobo_rows].copy()
            Svlad_lobo[same_burst[lobo_rows]] = -1e9
            Svlad_loio = S_vlad_gg.copy()
            Svlad_loio[same_id] = -1e9
            Cv_max_lobo, Cv_lse_lobo, _, _ = aggregate_class_matrices(Svlad_lobo, yg, u_cls, c_inv)
            Cv_max_loio, _, _, _ = aggregate_class_matrices(Svlad_loio, yg, u_cls, c_inv)
            Cv_max_val, Cv_lse_val, _, _ = aggregate_class_matrices(S_vlad_qg, yg, u_cls, c_inv)

            # Fast Leave-Burst-Out Ridge on top-768 whitened + LDA (~600 dims -> 0.08s invert!)
            d_keep = min(640, Zg_w.shape[1])
            X_clf_g = np.hstack([Zg_w[:, -d_keep:], 0.5 * Lg])
            X_clf_q = np.hstack([Zq_w[:, -d_keep:], 0.5 * Lq])
            Y_onehot = np.zeros((len(yg), len(u_cls)), dtype=np.float32)
            Y_onehot[np.arange(len(yg)), c_inv] = 1.0
            lam = 6.0
            A_inv = np.linalg.inv(X_clf_g.T @ X_clf_g + lam * np.eye(X_clf_g.shape[1], dtype=np.float32))
            W_ridge = A_inv @ (X_clf_g.T @ Y_onehot)
            C_ridge_val = X_clf_q @ W_ridge

            C_ridge_lobo = np.zeros((len(lobo_rows), len(u_cls)), dtype=np.float32)
            H_mat = X_clf_g @ A_inv @ X_clf_g.T
            full_pred_g = X_clf_g @ W_ridge
            for r_i, g_i in enumerate(lobo_rows):
                b_idx = np.where(gal_bursts == gal_bursts[g_i])[0]
                if len(b_idx) == 1:
                    h_ii = H_mat[g_i, g_i]
                    C_ridge_lobo[r_i] = (full_pred_g[g_i] - h_ii * Y_onehot[g_i]) / max(1e-3, 1.0 - h_ii)
                else:
                    H_bb = H_mat[np.ix_(b_idx, b_idx)]
                    corr = H_mat[g_i, b_idx] @ np.linalg.solve(
                        np.eye(len(b_idx), dtype=np.float32) - H_bb + 1e-4 * np.eye(len(b_idx), dtype=np.float32),
                        Y_onehot[b_idx] - full_pred_g[b_idx]
                    )
                    C_ridge_lobo[r_i] = full_pred_g[g_i] - corr

            def norm_rows(M):
                return (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-5) * 0.12

            Cr_lobo = norm_rows(C_ridge_lobo)
            Cr_val = norm_rows(C_ridge_val)

            # Query-to-Query cluster consolidation
            S_qq = Zq_w @ Zq_w.T
            W_qq = np.where((S_qq >= 0.68) & (~np.eye(len(Zq_w), dtype=bool)), np.maximum(0.0, S_qq) ** 4, 0.0)
            row_sum_qq = W_qq.sum(axis=1, keepdims=True)
            has_qq = (row_sum_qq[:, 0] > 0)

            rank_dict = {
                "B1_Decoupled_WhitenOpen_LSE_Rank": (C_lse_w_lobo, C_lse_w_val, None, None, None),
                "B2_Decoupled_WhitenOpen_LSE_IdCSLS": (identity_level_csls(C_lse_w_lobo), identity_level_csls(C_lse_w_val), None, None, None),
                "B3_Decoupled_Plus_ShrinkageLDA": (
                    0.65 * identity_level_csls(C_lse_w_lobo) + 0.35 * identity_level_csls(C_lse_lda_lobo),
                    0.65 * identity_level_csls(C_lse_w_val) + 0.35 * identity_level_csls(C_lse_lda_val),
                    None, None, None
                ),
            }
            rk_b3_lobo, rk_b3_val = rank_dict["B3_Decoupled_Plus_ShrinkageLDA"][:2]
            rk_b4_lobo = rk_b3_lobo + 0.35 * Cr_lobo
            rk_b4_val = rk_b3_val + 0.35 * Cr_val
            rank_dict["B4_Decoupled_LDA_Plus_RidgeClf"] = (rk_b4_lobo, rk_b4_val, None, None, None)

            rk_b5_lobo = rk_b4_lobo + 0.15 * identity_level_csls(Cv_lse_lobo)
            rk_b5_val = rk_b4_val + 0.15 * identity_level_csls(Cv_lse_val)
            rank_dict["B5_Decoupled_LDA_Ridge_Plus_VLAD"] = (
                rk_b5_lobo, rk_b5_val,
                Cv_max_lobo.max(axis=1, keepdims=True), Cv_max_loio.max(axis=1, keepdims=True), Cv_max_val.max(axis=1, keepdims=True)
            )

            rk_b6_val = rk_b5_val.copy()
            op_b6_val = C_open_val.copy()
            if np.any(has_qq):
                rk_b6_val[has_qq] = 0.72 * rk_b5_val[has_qq] + 0.28 * ((W_qq[has_qq] @ rk_b5_val) / row_sum_qq[has_qq])
                op_b6_val[has_qq] = 0.65 * C_open_val[has_qq] + 0.35 * ((W_qq[has_qq] @ C_open_val) / row_sum_qq[has_qq])

            rank_dict["B6_Decoupled_All_Plus_QueryClusterQE"] = (
                rk_b5_lobo, rk_b6_val,
                Cv_max_lobo.max(axis=1, keepdims=True), Cv_max_loio.max(axis=1, keepdims=True), Cv_max_val.max(axis=1, keepdims=True)
            )

            if has_sift_cache:
                sb_lobo, sex_lobo = lookup_sift_bonus_from_cache(
                    g_rows[lobo_rows], g_rows, c_inv, len(u_cls), ~same_burst[lobo_rows], sift_nbr_idx, sift_nbr_feats
                )
                _, sex_loio = lookup_sift_bonus_from_cache(
                    g_rows, g_rows, c_inv, len(u_cls), ~same_id, sift_nbr_idx, sift_nbr_feats
                )
                sb_val, sex_val = lookup_sift_bonus_from_cache(
                    v_rows, g_rows, c_inv, len(u_cls), np.ones((len(v_rows), len(g_rows)), dtype=bool), sift_nbr_idx, sift_nbr_feats
                )
                rk_b7_lobo = rk_b5_lobo + sb_lobo
                rk_b7_val_pre = rk_b5_val + sb_val
                rk_b7_val = rk_b7_val_pre.copy()
                if np.any(has_qq):
                    rk_b7_val[has_qq] = 0.72 * rk_b7_val_pre[has_qq] + 0.28 * ((W_qq[has_qq] @ rk_b7_val_pre) / row_sum_qq[has_qq])
                rank_dict["B7_Decoupled_All_Plus_SIFT_RANSAC"] = (
                    rk_b7_lobo, rk_b7_val,
                    np.hstack([Cv_max_lobo.max(axis=1, keepdims=True), sex_lobo]),
                    np.hstack([Cv_max_loio.max(axis=1, keepdims=True), sex_loio]),
                    np.hstack([Cv_max_val.max(axis=1, keepdims=True), sex_val]),
                )

            for cname in [c[0] for c in configs[4:]]:
                rk_lobo, rk_val, ex_lobo, ex_loio, ex_val = rank_dict[cname]
                op_val_use = op_b6_val if cname in ("B6_Decoupled_All_Plus_QueryClusterQE", "B7_Decoupled_All_Plus_SIFT_RANSAC") else C_open_val
                preds = decoupled_calibrate_and_predict(
                    rk_lobo, C_open_lobo, true_cls_lobo, C_open_loio,
                    rk_val, op_val_use, u_cls, EST_PI_NEW[sp],
                    extra_open_lobo=ex_lobo, extra_open_loio=ex_loio, extra_open_val=ex_val
                )
                for p_i, pr in zip(v_pos, preds):
                    fold_preds_by_cfg[cname][fk][p_i] = pr

        print(f"Finished Fold {fk} in {time.time() - tf:.1f}s")

    print(f"\nAll 5 folds completed in {time.time() - t0:.1f}s! Evaluating...")
    rows = []
    for cname, _ in configs:
        print(f"\n--- [{cname}] ---")
        res = evaluate_cv_predictions(cv, train_df, fold_preds_by_cfg[cname], verbose=True)
        res["name"] = cname
        rows.append(res)

    df_out = pd.DataFrame(rows)[["name", "lb_proxy_map5", "hyena_proxy_map5", "leopard_proxy_map5", "hyena_known_map5", "hyena_new_map5", "leopard_known_map5", "leopard_new_map5"]]
    print("\n================ EXP02 SUMMARY TABLE ================")
    print(df_out.sort_values("lb_proxy_map5", ascending=False).to_string(index=False))
    df_out.to_csv("cache/exp02_results.csv", index=False)


if __name__ == "__main__":
    run_suite()
