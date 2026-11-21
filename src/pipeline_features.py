import os
import time
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


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


def build_internal_subfolds(gal_global_rows, gal_identities, X_norm_gal, n_splits=4, seed=101):
    rng = np.random.RandomState(seed)
    bursts = compute_burst_ids(X_norm_gal, gal_identities, thresh=0.875)
    df_tmp = pd.DataFrame({
        "loc": np.arange(len(gal_global_rows)),
        "identity": gal_identities,
        "burst": bursts,
    })
    id_stats = df_tmp.groupby("identity").agg(n_rows=("loc", "count"), n_bursts=("burst", "nunique")).reset_index()
    id_stats["jitter"] = rng.uniform(0, 1, size=len(id_stats))
    id_stats = id_stats.sort_values(["n_bursts", "n_rows", "jitter"], ascending=[False, False, True]).reset_index(drop=True)

    unseen_map = {}
    for i, r in id_stats.iterrows():
        blk, pos = i // n_splits, i % n_splits
        unseen_map[r["identity"]] = int(pos if blk % 2 == 0 else (n_splits - 1 - pos))

    enc_fold = np.full(len(gal_global_rows), -1, dtype=np.int32)
    for ident, grp in df_tmp.groupby("identity"):
        b_sizes = grp.groupby("burst").size().sort_values(ascending=False)
        b_list = list(b_sizes.index)
        if len(b_list) < 2:
            continue
        uf = unseen_map[ident]
        others = [f for f in range(n_splits) if f != uf]
        rng.shuffle(others)
        cyc = others if len(b_list) <= len(others) else (others + [uf])
        for j, bid in enumerate(b_list):
            enc_fold[grp.index[grp["burst"] == bid]] = int(cyc[j % len(cyc)])

    subfolds = []
    for k in range(n_splits):
        sg, sq, slbl, snew = [], [], [], []
        for i in range(len(gal_global_rows)):
            ident = gal_identities[i]
            if unseen_map[ident] == k:
                sq.append(i)
                slbl.append("new_individual")
                snew.append(True)
            elif enc_fold[i] == k:
                sq.append(i)
                slbl.append(ident)
                snew.append(False)
            else:
                sg.append(i)
        subfolds.append({
            "sub_fold": k,
            "sub_g_loc": np.array(sg, dtype=np.int32),
            "sub_q_loc": np.array(sq, dtype=np.int32),
            "sub_q_labels": slbl,
            "sub_q_is_new": np.array(snew, dtype=bool),
        })
    return subfolds


def precompute_pca_basis(Xg, Xq, signed_pow=0.85):
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
        proj = U[:, m] * np.sqrt(N * eigvals)[None, :]
    else:
        C = (Xc.T @ Xc) / N
        eigvals, V = np.linalg.eigh(C)
        m = eigvals > 1e-8
        eigvals = eigvals[m]
        proj = Xc @ V[:, m]
    return proj, eigvals, len(Xg), Xg_n.astype(np.float32), Xq_n.astype(np.float32)


def slice_whiten(proj, eigvals, n_g, power, reg, top_d):
    d = min(top_d, proj.shape[1])
    p_sub = proj[:, -d:]
    e_sub = eigvals[-d:]
    Z = normalize(p_sub * ((e_sub + reg) ** (-0.5 * power))[None, :], axis=1).astype(np.float32)
    return Z[:n_g], Z[n_g:]


def fast_wccn(Zg, cols_per_class, Zq, reg=0.06, power=0.50):
    """Fast Within-Class Covariance Normalization on reduced whitened basis (~0.04s)."""
    res_list = [Zg[cols] - Zg[cols].mean(axis=0, keepdims=True) for cols in cols_per_class if len(cols) >= 2]
    R = np.vstack(res_list)
    C_w = (R.T @ R) / float(len(R))
    eig, V = np.linalg.eigh(C_w)
    scale = (np.maximum(0.0, eig) + reg) ** (-power)
    W = (V * scale[None, :]) @ V.T
    return normalize(Zg @ W, axis=1).astype(np.float32), normalize(Zq @ W, axis=1).astype(np.float32)


def apply_joint_alpha_qe(Zg, Zq, top_k=4, alpha=3.2, sim_floor=0.22):
    Z_all = np.vstack([Zg, Zq])
    S = Z_all @ Z_all.T
    part = np.argpartition(-S, kth=min(top_k, len(Z_all) - 1), axis=1)[:, :top_k]
    s_top = np.take_along_axis(S, part, axis=1)
    w = np.where(s_top >= sim_floor, np.maximum(0.0, s_top) ** alpha, 0.0)
    Z_nbr = np.sum(w[:, :, None] * Z_all[part], axis=1)
    Z_out = normalize(Z_all + Z_nbr, axis=1).astype(np.float32)
    return Z_out[:len(Zg)], Z_out[len(Zg):]


def compute_knn_jaccard(Zg, Zq, k=16):
    """
    Fast rank-based Mutual k-NN Jaccard overlap (n_q, n_g) via sparse binary indicator GEMM!
    100% scale-invariant to gallery size!
    """
    Z_all = np.vstack([Zg, Zq])
    N = len(Z_all)
    n_g = len(Zg)
    S = Z_all @ Z_all.T
    kk = min(k, N - 1)
    nbrs = np.argpartition(-S, kth=kk, axis=1)[:, :kk]  # (N, kk)
    A = np.zeros((N, N), dtype=np.float32)
    A[np.arange(N)[:, None], nbrs] = 1.0
    # Mutual / symmetric neighborhood indicator
    A_sym = np.minimum(1.0, 0.5 * (A + A.T) + 0.5 * (A * A.T))
    inter = A_sym[n_g:] @ A_sym[:n_g].T  # (n_q, n_g)
    deg_q = A_sym[n_g:].sum(axis=1, keepdims=True)
    deg_g = A_sym[:n_g].sum(axis=1)[None, :]
    return inter / np.maximum(1.0, deg_q + deg_g - inter)


def fast_class_stats(S_qg, cols_per_class):
    n_q = S_qg.shape[0]
    n_c = len(cols_per_class)
    C_max = np.empty((n_q, n_c), dtype=np.float32)
    C_top2 = np.empty((n_q, n_c), dtype=np.float32)
    C_mean = np.empty((n_q, n_c), dtype=np.float32)
    best_g = np.empty((n_q, n_c), dtype=np.int32)

    for c, cols in enumerate(cols_per_class):
        sub = S_qg[:, cols]
        arg_m = np.argmax(sub, axis=1)
        mx = sub[np.arange(n_q), arg_m]
        C_max[:, c] = mx
        best_g[:, c] = cols[arg_m]
        C_mean[:, c] = sub.mean(axis=1)
        if len(cols) == 1:
            C_top2[:, c] = mx
        else:
            p2 = np.partition(sub, kth=len(cols) - 2, axis=1)[:, -2:]
            C_top2[:, c] = 0.5 * (p2[:, 0] + p2[:, 1])
    return C_max, C_top2, C_mean, best_g


def loio_gallery_calibration_features(S_gg_masked, q_s1_vec):
    """
    Computes EXACT Gallery-Size-Invariant Open-Set Z-score & Empirical CDF!
    S_gg_masked has same-identity entries set to -2.0, so S_gg_masked.max(axis=1)
    is the exact distribution of max impostor similarity against THIS gallery of size n_g!
    """
    loio_max = np.sort(S_gg_masked.max(axis=1))
    mu = float(loio_max.mean())
    sd = float(loio_max.std()) + 1e-4
    q90 = float(np.quantile(loio_max, 0.90))
    z_loio = (q_s1_vec - mu) / sd
    diff_q90 = q_s1_vec - q90
    ecdf = np.searchsorted(loio_max, q_s1_vec).astype(np.float32) / float(len(loio_max))
    return z_loio.astype(np.float32), diff_q90.astype(np.float32), ecdf


CAND_FEATURE_NAMES = [
    "base_ret", "diff_base",
    "p55_max", "p55_top2", "p55_mean", "p55_cent", "p55_csls", "diff_p55", "z_p55_k", "p55_z_loio",
    "p68_max", "p68_top2", "diff_p68",
    "raw_max", "diff_raw",
    "lda_max", "lda_top2", "lda_cent", "lda_csls", "diff_lda", "z_lda_k", "lda_z_loio",
    "wccn_max", "diff_wccn",
    "qe_max", "qe_top2", "diff_qe",
    "jacc_max", "diff_jacc",
    "vl_max", "vl_top2", "diff_vl",
    "ridge_z", "diff_ridge",
    "qq_p55", "diff_qq_p55",
    "sift_inl", "sift_suminl", "sift_r72", "sift_mut", "sift_marg", "sift_rat", "diff_sift_inl",
    "vp_has_same", "vp_same_frac", "vp_same_max", "vp_diff_max", "vp_soft_best",
    "cand_log_freq", "cand_hub", "rank_inv"
]


def build_feature_bundle(
    g_rows, q_rows, species,
    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, kp_counts,
    sift_nbr_idx, sift_nbr_feats, top_k_cands=12
):
    n_g, n_q = len(g_rows), len(q_rows)
    yg = all_identities[g_rows]
    u_cls, c_inv = np.unique(yg, return_inverse=True)
    n_c = len(u_cls)
    cols_per_class = [np.where(c_inv == c)[0] for c in range(n_c)]
    class_counts = np.array([len(cols) for cols in cols_per_class], dtype=np.float32)
    log_class_freq = np.log((class_counts + 0.5) / float(n_g))

    Xg_raw, Xq_raw = miew_all[g_rows], miew_all[q_rows]
    proj85, eig85, _, Xg_n85, Xq_n85 = precompute_pca_basis(Xg_raw, Xq_raw, signed_pow=0.85)

    d_mid = 640 if species == "hyena" else 768
    d_hi = 900 if species == "hyena" else 1152
    Zg_p55, Zq_p55 = slice_whiten(proj85, eig85, n_g, power=0.55, reg=1.5e-3, top_d=d_mid)
    Zg_p68, Zq_p68 = slice_whiten(proj85, eig85, n_g, power=0.68, reg=2.5e-3, top_d=d_hi)

    mu_raw = np.vstack([Xg_n85, Xq_n85]).mean(axis=0, keepdims=True)
    Zg_rawc = normalize(Xg_n85 - mu_raw, axis=1).astype(np.float32)
    Zq_rawc = normalize(Xq_n85 - mu_raw, axis=1).astype(np.float32)

    d_lda_in = min(512, Zg_p55.shape[1])
    Zg_lin = normalize(Zg_p55[:, -d_lda_in:], axis=1)
    Zq_lin = normalize(Zq_p55[:, -d_lda_in:], axis=1)
    valid_cls_mask = class_counts[c_inv] >= 2
    n_valid_cls = int((class_counts >= 2).sum())
    d_lda_out = min(220, max(2, n_valid_cls - 1), d_lda_in)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.22, n_components=d_lda_out)
    lda.fit(Zg_lin[valid_cls_mask], yg[valid_cls_mask])
    Lg = normalize(lda.transform(Zg_lin), axis=1).astype(np.float32)
    Lq = normalize(lda.transform(Zq_lin), axis=1).astype(np.float32)

    # WCCN full-rank metric projection on top-512 whitened basis
    Wg_wccn, Wq_wccn = fast_wccn(Zg_lin, cols_per_class, Zq_lin, reg=0.06, power=0.50)

    # Joint Transductive Alpha-QE + Mutual k-NN Jaccard
    Zg_mix = normalize(np.hstack([0.80 * Zg_p55, 0.60 * Lg]), axis=1)
    Zq_mix = normalize(np.hstack([0.80 * Zq_p55, 0.60 * Lq]), axis=1)
    Zg_qe, Zq_qe = apply_joint_alpha_qe(Zg_mix, Zq_mix, top_k=4, alpha=3.2, sim_floor=0.25)
    S_jacc = compute_knn_jaccard(Zg_mix, Zq_mix, k=16)

    # Whitened RootSIFT VLAD
    proj_vl, eig_vl, _, _, _ = precompute_pca_basis(vlad_all[g_rows], vlad_all[q_rows], signed_pow=1.0)
    Vg_vl, Vq_vl = slice_whiten(proj_vl, eig_vl, n_g, power=0.40, reg=1e-2, top_d=512)

    # Multi-class Closed-Form Ridge Classifier
    X_clf_g = np.hstack([Zg_lin, 0.5 * Lg])
    X_clf_q = np.hstack([Zq_lin, 0.5 * Lq])
    Y_oh = np.zeros((n_g, n_c), dtype=np.float32)
    Y_oh[np.arange(n_g), c_inv] = 1.0
    W_ridge = np.linalg.solve(
        X_clf_g.T @ X_clf_g + 6.5 * np.eye(X_clf_g.shape[1], dtype=np.float32),
        X_clf_g.T @ Y_oh
    )
    C_ridge = X_clf_q @ W_ridge
    C_ridge_z = (C_ridge - C_ridge.mean(axis=1, keepdims=True)) / (C_ridge.std(axis=1, keepdims=True) + 1e-5)

    # Soft Viewpoint Predictor
    vp_g = all_viewpoints[g_rows]
    vp_q = all_viewpoints[q_rows]
    known_vp_mask = (vp_g != "MISSING")
    vp_clf = LogisticRegression(C=2.0, max_iter=300)
    vp_clf.fit(Zg_rawc[known_vp_mask, :512], vp_g[known_vp_mask])
    P_vp_g = vp_clf.predict_proba(Zg_rawc[:, :512]).astype(np.float32)
    P_vp_q = vp_clf.predict_proba(Zq_rawc[:, :512]).astype(np.float32)
    eff_vp_g = np.where(vp_g != "MISSING", vp_g, vp_clf.classes_[np.argmax(P_vp_g, axis=1)])
    eff_vp_q = np.where(vp_q != "MISSING", vp_q, vp_clf.classes_[np.argmax(P_vp_q, axis=1)])

    # Pairwise Similarity Matrices
    S_p55 = Zq_p55 @ Zg_p55.T
    S_p68 = Zq_p68 @ Zg_p68.T
    S_raw = Zq_rawc @ Zg_rawc.T
    S_lda = Lq @ Lg.T
    S_wccn = Wq_wccn @ Wg_wccn.T
    S_qe = Zq_qe @ Zg_qe.T
    S_vl = Vq_vl @ Vg_vl.T

    # Gallery LOIO impostor distributions (for exact gallery-size invariance!)
    same_id_gg = (c_inv[:, None] == c_inv[None, :])
    S_gg_p55 = Zg_p55 @ Zg_p55.T
    S_gg_p55[same_id_gg] = -2.0
    S_gg_p68 = Zg_p68 @ Zg_p68.T
    S_gg_p68[same_id_gg] = -2.0
    S_gg_raw = Zg_rawc @ Zg_rawc.T
    S_gg_raw[same_id_gg] = -2.0
    S_gg_lda = Lg @ Lg.T
    S_gg_lda[same_id_gg] = -2.0

    loio_p55_mu = float(S_gg_p55.max(axis=1).mean())
    loio_p55_sd = float(S_gg_p55.max(axis=1).std()) + 1e-4
    loio_lda_mu = float(S_gg_lda.max(axis=1).mean())
    loio_lda_sd = float(S_gg_lda.max(axis=1).std()) + 1e-4

    gal_imp_hub_img = np.partition(S_gg_p55, kth=max(0, n_g - 16), axis=1)[:, -15:].mean(axis=1)
    class_imp_hub_p55 = np.array([gal_imp_hub_img[cols].max() for cols in cols_per_class], dtype=np.float32)

    cent_p55 = normalize(np.vstack([Zg_p55[cols].mean(axis=0) for cols in cols_per_class]), axis=1).astype(np.float32)
    cent_lda = normalize(np.vstack([Lg[cols].mean(axis=0) for cols in cols_per_class]), axis=1).astype(np.float32)
    C_cent_p55 = Zq_p55 @ cent_p55.T
    C_cent_lda = Lq @ cent_lda.T

    Cp55_max, Cp55_top2, Cp55_mean, best_g_p55 = fast_class_stats(S_p55, cols_per_class)
    Cp68_max, Cp68_top2, _, _ = fast_class_stats(S_p68, cols_per_class)
    Craw_max, _, _, _ = fast_class_stats(S_raw, cols_per_class)
    Clda_max, Clda_top2, _, _ = fast_class_stats(S_lda, cols_per_class)
    Cwccn_max, _, _, _ = fast_class_stats(S_wccn, cols_per_class)
    Cqe_max, Cqe_top2, _, _ = fast_class_stats(S_qe, cols_per_class)
    Cjacc_max, _, _, _ = fast_class_stats(S_jacc, cols_per_class)
    Cvl_max, Cvl_top2, _, _ = fast_class_stats(S_vl, cols_per_class)

    # Vectorized O(1) SIFT + 4-DOF RANSAC Lookup
    g_to_loc = np.full(len(miew_all), -1, dtype=np.int32)
    g_to_loc[g_rows] = np.arange(n_g, dtype=np.int32)
    q_nbr_loc = g_to_loc[sift_nbr_idx[q_rows]]
    q_sift_f = sift_nbr_feats[q_rows]
    valid_sift = (q_nbr_loc >= 0)

    Csift_max_inl = np.zeros((n_q, n_c), dtype=np.float32)
    Csift_sum_inl = np.zeros((n_q, n_c), dtype=np.float32)
    Csift_max_r72 = np.zeros((n_q, n_c), dtype=np.float32)
    Csift_max_mut = np.zeros((n_q, n_c), dtype=np.float32)
    Csift_margin = np.zeros((n_q, n_c), dtype=np.float32)
    Csift_inl_rat = np.zeros((n_q, n_c), dtype=np.float32)

    q_idx_nz, m_idx_nz = np.where(valid_sift)
    cls_nz = c_inv[q_nbr_loc[q_idx_nz, m_idx_nz]]
    f_nz = q_sift_f[q_idx_nz, m_idx_nz]

    np.maximum.at(Csift_max_r72, (q_idx_nz, cls_nz), f_nz[:, 1])
    np.maximum.at(Csift_max_mut, (q_idx_nz, cls_nz), f_nz[:, 2])
    np.maximum.at(Csift_margin, (q_idx_nz, cls_nz), f_nz[:, 3])
    np.maximum.at(Csift_max_inl, (q_idx_nz, cls_nz), f_nz[:, 4])
    np.add.at(Csift_sum_inl, (q_idx_nz, cls_nz), f_nz[:, 4])
    np.maximum.at(Csift_inl_rat, (q_idx_nz, cls_nz), f_nz[:, 5])

    sift_bonus = 0.014 * np.log1p(Csift_max_inl) + 0.007 * np.log1p(Csift_max_r72) + 0.025 * (Csift_max_inl >= 8)

    # Query-to-Query Cluster Consolidation
    S_qq_raw = Zq_rawc @ Zq_rawc.T
    S_qq_p55 = Zq_p55 @ Zq_p55.T
    np.fill_diagonal(S_qq_raw, -2.0)
    np.fill_diagonal(S_qq_p55, -2.0)
    max_qq_raw = S_qq_raw.max(axis=1)
    max_qq_p55 = S_qq_p55.max(axis=1)
    n_qq_nbrs = (S_qq_raw >= 0.65).sum(axis=1).astype(np.float32)

    W_qq = np.where(S_qq_raw >= 0.65, np.maximum(0.0, S_qq_raw) ** 4, 0.0)
    w_qq_sum = W_qq.sum(axis=1, keepdims=True)
    has_qq = (w_qq_sum[:, 0] > 0)
    C_qq_p55 = Cp55_max.copy()
    if np.any(has_qq):
        C_qq_p55[has_qq] = 0.65 * Cp55_max[has_qq] + 0.35 * ((W_qq[has_qq] @ Cp55_max) / w_qq_sum[has_qq])

    def id_csls(M):
        bg = -np.sort(-M, axis=1)[:, 1:min(16, n_c)].mean(axis=1, keepdims=True)
        return M - 0.5 * bg

    Cp55_csls = id_csls(Cp55_max) - 0.35 * class_imp_hub_p55[None, :]
    Clda_csls = id_csls(Clda_max)
    Cqe_csls = id_csls(Cqe_max)

    C_base_retrieval = (
        0.34 * Cp55_csls
        + 0.12 * id_csls(Cp55_top2)
        + 0.18 * Clda_csls
        + 0.08 * id_csls(Cwccn_max)
        + 0.13 * Cqe_csls
        + 0.05 * Cjacc_max
        + 0.05 * id_csls(Cvl_max)
        + 0.05 * (0.12 * C_ridge_z)
        + sift_bonus
    )

    K = min(top_k_cands, n_c)
    top_cands = np.argsort(-C_base_retrieval, axis=1)[:, :K]
    rows_2d = np.arange(n_q)[:, None]

    def gather(M):
        return M[rows_2d, top_cands]

    def best_other_diff(V_k):
        idx2 = np.argsort(-V_k, axis=1)[:, :2]
        v1 = V_k[np.arange(n_q), idx2[:, 0]]
        v2 = V_k[np.arange(n_q), idx2[:, 1]]
        is_best = (np.arange(K)[None, :] == idx2[:, :1])
        ref = np.where(is_best, v2[:, None], v1[:, None])
        return V_k - ref

    g_p55_max = gather(Cp55_max)
    g_p55_top2 = gather(Cp55_top2)
    g_p55_mean = gather(Cp55_mean)
    g_p55_cent = gather(C_cent_p55)
    g_p55_csls = gather(Cp55_csls)
    g_p68_max = gather(Cp68_max)
    g_p68_top2 = gather(Cp68_top2)
    g_raw_max = gather(Craw_max)
    g_lda_max = gather(Clda_max)
    g_lda_top2 = gather(Clda_top2)
    g_lda_cent = gather(C_cent_lda)
    g_lda_csls = gather(Clda_csls)
    g_wccn_max = gather(Cwccn_max)
    g_qe_max = gather(Cqe_max)
    g_qe_top2 = gather(Cqe_top2)
    g_jacc_max = gather(Cjacc_max)
    g_vl_max = gather(Cvl_max)
    g_vl_top2 = gather(Cvl_top2)
    g_ridge_z = gather(C_ridge_z)
    g_qq_p55 = gather(C_qq_p55)
    g_base = gather(C_base_retrieval)

    g_sift_inl = np.log1p(gather(Csift_max_inl))
    g_sift_suminl = np.log1p(gather(Csift_sum_inl))
    g_sift_r72 = np.log1p(gather(Csift_max_r72))
    g_sift_mut = np.log1p(gather(Csift_max_mut))
    g_sift_marg = gather(Csift_margin)
    g_sift_rat = gather(Csift_inl_rat)

    vp_same_max = np.full((n_q, K), -0.15, dtype=np.float32)
    vp_diff_max = np.full((n_q, K), -0.15, dtype=np.float32)
    vp_same_frac = np.zeros((n_q, K), dtype=np.float32)
    vp_has_same = np.zeros((n_q, K), dtype=np.float32)

    best_g_cand = gather(best_g_p55)
    vp_soft_best = np.sum(P_vp_q[:, None, :] * P_vp_g[best_g_cand], axis=2)

    for qi in range(n_q):
        vq_str = eff_vp_q[qi]
        s_row = S_p55[qi]
        for ki in range(K):
            cid = top_cands[qi, ki]
            cols = cols_per_class[cid]
            same_m = (eff_vp_g[cols] == vq_str)
            ns = int(same_m.sum())
            if ns > 0:
                vp_has_same[qi, ki] = 1.0
                vp_same_frac[qi, ki] = float(ns) / len(cols)
                vp_same_max[qi, ki] = s_row[cols[same_m]].max()
            if ns < len(cols):
                vp_diff_max[qi, ki] = s_row[cols[~same_m]].max()

    cand_log_freq = log_class_freq[top_cands]
    cand_hub = class_imp_hub_p55[top_cands]
    rank_inv = (1.0 / (1.0 + np.arange(K, dtype=np.float32)))[None, :].repeat(n_q, axis=0)

    z_p55_in_k = (g_p55_max - g_p55_max.mean(axis=1, keepdims=True)) / (g_p55_max.std(axis=1, keepdims=True) + 1e-4)
    z_lda_in_k = (g_lda_max - g_lda_max.mean(axis=1, keepdims=True)) / (g_lda_max.std(axis=1, keepdims=True) + 1e-4)

    X_cand = np.stack([
        g_base, best_other_diff(g_base),
        g_p55_max, g_p55_top2, g_p55_mean, g_p55_cent, g_p55_csls, best_other_diff(g_p55_max), z_p55_in_k, (g_p55_max - loio_p55_mu) / loio_p55_sd,
        g_p68_max, g_p68_top2, best_other_diff(g_p68_max),
        g_raw_max, best_other_diff(g_raw_max),
        g_lda_max, g_lda_top2, g_lda_cent, g_lda_csls, best_other_diff(g_lda_max), z_lda_in_k, (g_lda_max - loio_lda_mu) / loio_lda_sd,
        g_wccn_max, best_other_diff(g_wccn_max),
        g_qe_max, g_qe_top2, best_other_diff(g_qe_max),
        g_jacc_max, best_other_diff(g_jacc_max),
        g_vl_max, g_vl_top2, best_other_diff(g_vl_max),
        g_ridge_z, best_other_diff(g_ridge_z),
        g_qq_p55, best_other_diff(g_qq_p55),
        g_sift_inl, g_sift_suminl, g_sift_r72, g_sift_mut, g_sift_marg, g_sift_rat, best_other_diff(g_sift_inl),
        vp_has_same, vp_same_frac, vp_same_max, vp_diff_max, vp_soft_best,
        cand_log_freq, cand_hub, rank_inv,
    ], axis=2).astype(np.float32)  # (n_q, K, 51)

    # Query-Level Open-Set Feature Matrix X_open_base (with EXACT LOIO gallery-size-invariant calibration!)
    def top_profile(M):
        s = -np.sort(-M, axis=1)[:, :10]
        s1, s2, s3, s5 = s[:, 0], s[:, 1], s[:, 2], s[:, 4]
        bg = s[:, 1:10].mean(axis=1)
        std = s[:, 1:10].std(axis=1) + 1e-4
        return [s1, s2, s1 - s2, s1 - s3, s1 - s5, bg, (s1 - bg) / std]

    p55_s1 = -np.sort(-Cp55_max, axis=1)[:, 0]
    p68_s1 = -np.sort(-Cp68_max, axis=1)[:, 0]
    raw_s1 = -np.sort(-Craw_max, axis=1)[:, 0]
    lda_s1 = -np.sort(-Clda_max, axis=1)[:, 0]

    z_l55, dq90_55, ecdf_55 = loio_gallery_calibration_features(S_gg_p55, p55_s1)
    z_l68, dq90_68, ecdf_68 = loio_gallery_calibration_features(S_gg_p68, p68_s1)
    z_lraw, dq90_raw, ecdf_raw = loio_gallery_calibration_features(S_gg_raw, raw_s1)
    z_llda, dq90_lda, ecdf_lda = loio_gallery_calibration_features(S_gg_lda, lda_s1)

    cluster_max_z55 = z_l55.copy()
    cluster_mean_z55 = z_l55.copy()
    max_sift_inl_any = np.log1p(Csift_max_inl.max(axis=1))
    cluster_max_sift_inl = max_sift_inl_any.copy()

    for qi in range(n_q):
        nbr_m = (S_qq_raw[qi] >= 0.68)
        if np.any(nbr_m):
            vals = np.append(z_l55[nbr_m], z_l55[qi])
            cluster_max_z55[qi] = vals.max()
            cluster_mean_z55[qi] = vals.mean()
            cluster_max_sift_inl[qi] = max(max_sift_inl_any[qi], max_sift_inl_any[nbr_m].max())

    open_cols = (
        top_profile(Cp55_max)
        + top_profile(Cp68_max)
        + top_profile(Craw_max)
        + top_profile(Clda_max)
        + top_profile(Cwccn_max)[:4]
        + top_profile(Cqe_max)[:5]
        + top_profile(Cjacc_max)[:3]
        + top_profile(Cvl_max)[:4]
        + [
            z_l55, dq90_55, ecdf_55,
            z_l68, dq90_68, ecdf_68,
            z_lraw, dq90_raw, ecdf_raw,
            z_llda, dq90_lda, ecdf_lda,
            C_ridge_z.max(axis=1),
            -np.sort(-C_ridge_z, axis=1)[:, 0] - (-np.sort(-C_ridge_z, axis=1)[:, 1]),
            max_sift_inl_any,
            np.log1p(-np.sort(-Csift_max_inl, axis=1)[:, 1]),
            g_sift_inl[:, 0],
            np.log1p(Csift_max_r72.max(axis=1)),
            np.log1p(Csift_max_mut.max(axis=1)),
            vp_has_same[:, 0],
            vp_same_frac[:, 0],
            vp_same_max[:, 0],
            vp_diff_max[:, 0],
            vp_soft_best[:, 0],
            cand_log_freq[:, 0],
            cand_hub[:, 0],
            np.log1p(kp_counts[q_rows].astype(np.float32)),
            miew_norms[q_rows].astype(np.float32),
            max_qq_raw,
            max_qq_p55,
            np.log1p(n_qq_nbrs),
            cluster_max_z55,
            cluster_mean_z55,
            cluster_max_sift_inl,
        ]
    )
    X_open_base = np.column_stack(open_cols).astype(np.float32)

    return {
        "u_cls": u_cls,
        "top_cands": top_cands,
        "cand_names": u_cls[top_cands],
        "X_cand": X_cand,              # (n_q, K, 51)
        "X_open_base": X_open_base,    # (n_q, 79)
        "W_qq": W_qq,
        "w_qq_sum": w_qq_sum,
        "has_qq": has_qq,
    }
