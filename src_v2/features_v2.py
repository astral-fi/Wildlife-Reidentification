import os
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# 1. Dual-SVD PCA Power-Whitening
# ---------------------------------------------------------------------------

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
    Z = normalize(proj[:, -d:] * ((eigvals[-d:] + reg) ** (-0.5 * power))[None, :], axis=1).astype(np.float32)
    return Z[:n_g], Z[n_g:]


# ---------------------------------------------------------------------------
# 2. Supervised Classical Metric Learning: WCCN, KISSME, Cross-View CCA
# ---------------------------------------------------------------------------

def fit_wccn(Zg, cols_per_class, Zq, reg=0.06, power=0.50):
    res = [Zg[cols] - Zg[cols].mean(axis=0, keepdims=True) for cols in cols_per_class if len(cols) >= 2]
    R = np.vstack(res)
    C_w = (R.T @ R) / float(len(R))
    eig, V = np.linalg.eigh(C_w)
    W = (V * ((np.maximum(0.0, eig) + reg) ** (-power))[None, :]) @ V.T
    return normalize(Zg @ W, axis=1).astype(np.float32), normalize(Zq @ W, axis=1).astype(np.float32)


def fit_kissme(Zg, cols_per_class, Zq, reg_s=0.12, reg_d=0.12, max_dim=256, seed=42):
    """
    KISSME (Kostinger et al., CVPR 2012):
      M = (Sigma_same + reg_s*I)^(-1) - (Sigma_diff + reg_d*I)^(-1), projected onto PSD cone!
    """
    d = min(max_dim, Zg.shape[1])
    Xg = normalize(Zg[:, -d:], axis=1).astype(np.float64)
    Xq = normalize(Zq[:, -d:], axis=1).astype(np.float64)
    rng = np.random.RandomState(seed)

    # Exact within-class difference covariance Sigma_same = 2 * Sigma_W (fast O(N*d^2) exact identity!)
    res_list = []
    for cols in cols_per_class:
        m = len(cols)
        if m >= 2:
            sub = Xg[cols]
            # Weight each class proportionally to sqrt(m) so 391-image class doesn't dominate Sigma_same
            w = np.sqrt(float(m)) / float(m)
            res_list.append((sub - sub.mean(0, keepdims=True)) * np.sqrt(w * m))
    R = np.vstack(res_list)
    Sigma_s = 2.0 * (R.T @ R) / float(len(R))

    # Exact between-class difference covariance via global covariance:
    # For random impostor pairs (x_i, x_j), E[(x_i - x_j)(x_i - x_j)^T] = 2 * Cov(Xg)
    Xc = Xg - Xg.mean(0, keepdims=True)
    Sigma_d = 2.0 * (Xc.T @ Xc) / float(len(Xg))

    inv_s = np.linalg.inv(Sigma_s + reg_s * np.eye(d))
    inv_d = np.linalg.inv(Sigma_d + reg_d * np.eye(d))
    M = 0.5 * ((inv_s - inv_d) + (inv_s - inv_d).T)
    eig, V = np.linalg.eigh(M)
    pos = eig > 1e-6
    if pos.sum() < 16:
        top_k = min(64, d)
        L = V[:, -top_k:]
    else:
        L = V[:, pos] * np.sqrt(eig[pos])[None, :]
    return normalize(Xg @ L, axis=1).astype(np.float32), normalize(Xq @ L, axis=1).astype(np.float32)


def fit_cross_view_cca(Zg, cols_per_class, eff_vp_g, Zq, eff_vp_q, reg=0.18, max_dim=224, n_comp=96):
    """
    Regularized Cross-View CCA between Left-flank and Right-flank sightings of multi-view animals!
    Learns the shared latent subspace bridging left <-> right flanks of the same individual.
    """
    left_set = {"left", "frontleft", "backleft"}
    right_set = {"right", "frontright", "backright"}

    d = min(max_dim, Zg.shape[1])
    Xg = normalize(Zg[:, -d:], axis=1).astype(np.float64)
    Xq = normalize(Zq[:, -d:], axis=1).astype(np.float64)

    L_protos, R_protos = [], []
    for cols in cols_per_class:
        vps = eff_vp_g[cols]
        l_idx = cols[np.isin(vps, list(left_set))]
        r_idx = cols[np.isin(vps, list(right_set))]
        if len(l_idx) >= 1 and len(r_idx) >= 1:
            L_protos.append(Xg[l_idx].mean(axis=0))
            R_protos.append(Xg[r_idx].mean(axis=0))

    if len(L_protos) < 20:
        return Zg[:, -n_comp:].astype(np.float32), Zq[:, -n_comp:].astype(np.float32)

    XL = np.vstack(L_protos)
    XR = np.vstack(R_protos)
    muL, muR = XL.mean(0, keepdims=True), XR.mean(0, keepdims=True)
    XLc, XRc = XL - muL, XR - muR
    n_p = float(len(XL))

    CLL = (XLc.T @ XLc) / n_p + reg * np.eye(d)
    CRR = (XRc.T @ XRc) / n_p + reg * np.eye(d)
    CLR = (XLc.T @ XRc) / n_p

    eL, vL = np.linalg.eigh(CLL)
    eR, vR = np.linalg.eigh(CRR)
    inv_sqrt_L = (vL * (np.maximum(eL, 1e-8) ** -0.5)[None, :]) @ vL.T
    inv_sqrt_R = (vR * (np.maximum(eR, 1e-8) ** -0.5)[None, :]) @ vR.T

    T_mat = inv_sqrt_L @ CLR @ inv_sqrt_R
    U, s_vals, Vt = np.linalg.svd(T_mat, full_matrices=False)
    k = min(n_comp, len(s_vals))
    # Weight canonical dimensions by sqrt(canonical correlation) so noisy tail correlations fade out
    corr_w = np.sqrt(np.clip(s_vals[:k], 0.0, 1.0))[None, :]
    WL = (inv_sqrt_L @ U[:, :k]) * corr_w
    WR = (inv_sqrt_R @ Vt[:k, :].T) * corr_w
    WM = 0.5 * (WL + WR)
    muM = 0.5 * (muL + muR)

    def project(X, vps):
        out = (X - muM) @ WM
        is_l = np.isin(vps, list(left_set))
        is_r = np.isin(vps, list(right_set))
        if np.any(is_l):
            out[is_l] = (X[is_l] - muL) @ WL
        if np.any(is_r):
            out[is_r] = (X[is_r] - muR) @ WR
        return normalize(out, axis=1).astype(np.float32)

    return project(Xg, eff_vp_g), project(Xq, eff_vp_q)


# ---------------------------------------------------------------------------
# 3. True Zhong et al. (CVPR 2017) k-Reciprocal Re-Ranking (Vectorized)
# ---------------------------------------------------------------------------

def zhong_k_reciprocal_sim(Z_gal, Z_qry, k1=18, k2=6):
    """
    Exact Zhong et al. (CVPR 2017) k-Reciprocal Re-Ranking Jaccard similarity (n_q, n_g)
    on joint gallery + query embeddings!
    Returns weighted Jaccard similarity in [0, 1] (higher = stronger reciprocal manifold match!).
    """
    Z_all = np.vstack([Z_gal, Z_qry])
    N, n_g = len(Z_all), len(Z_gal)
    S = np.clip(Z_all @ Z_all.T, -1.0, 1.0)
    D = np.maximum(0.0, 2.0 - 2.0 * S)  # squared Euclidean distance on unit sphere

    k1 = min(k1, N - 1)
    k2 = min(k2, N - 1)
    k1_half = max(2, int(round(k1 / 2.0)))

    # Initial sorted top-(k1+1) indices per node
    top_ext = np.argpartition(D, kth=min(k1 + 1, N - 1), axis=1)[:, :k1 + 1]
    d_ext = np.take_along_axis(D, top_ext, axis=1)
    ord_ext = np.argsort(d_ext, axis=1)
    initial_rank = np.take_along_axis(top_ext, ord_ext, axis=1)  # (N, k1+1)

    # Boolean matrix of top-(k1+1) forward neighbors for O(1) reciprocal checks
    fwd = np.zeros((N, N), dtype=bool)
    fwd[np.arange(N)[:, None], initial_rank] = True
    recip = fwd & fwd.T  # (N, N): exact k1-reciprocal relation!

    fwd_half = np.zeros((N, N), dtype=bool)
    fwd_half[np.arange(N)[:, None], initial_rank[:, :k1_half + 1]] = True
    recip_half = fwd_half & fwd_half.T
    recip_half_sizes = recip_half.sum(axis=1)

    # Expand k1-reciprocal sets via Zhong 2/3-overlap rule (100% sparse: ~80x faster & bit-exact!)
    R_sp = sparse.csr_matrix(recip.astype(np.float32))
    Rh_sp = sparse.csr_matrix(recip_half.astype(np.float32))
    overlap_sp = (R_sp @ Rh_sp.T).multiply(R_sp).tocoo()
    thresh = (2.0 / 3.0) * np.maximum(1.0, recip_half_sizes[overlap_sp.col])
    keep_e = overlap_sp.data > thresh
    EG_sp = sparse.csr_matrix(
        (np.ones(int(keep_e.sum()), dtype=np.float32), (overlap_sp.row[keep_e], overlap_sp.col[keep_e])),
        shape=(N, N),
    )
    R_star_sp = (R_sp + (EG_sp @ Rh_sp)).tocoo()
    R_star_sp.sum_duplicates()

    # Gaussian kernel weights V_ij = exp(-D_ij) for j in R*(i), row-L1 normalized (pure sparse!)
    v_vals = np.exp(-D[R_star_sp.row, R_star_sp.col]).astype(np.float32)
    V_sp = sparse.csr_matrix((v_vals, (R_star_sp.row, R_star_sp.col)), shape=(N, N))
    row_sums = np.asarray(V_sp.sum(axis=1)).ravel()
    V_sp = sparse.diags(1.0 / np.maximum(1e-8, row_sums).astype(np.float32)) @ V_sp

    # Local Query Expansion over top-k2 neighbors (sparse x sparse!)
    L_rows = np.repeat(np.arange(N, dtype=np.int32), k2)
    L_cols = initial_rank[:, :k2].ravel()
    L_mat = sparse.csr_matrix((np.full(len(L_rows), 1.0 / k2, dtype=np.float32), (L_rows, L_cols)), shape=(N, N))
    V_qe = (L_mat @ V_sp).tocsr()  # (N, N) sparse float32!

    # Exact Weighted Jaccard J(q, g) = M / (2 - M) where M = sum(min(Vq, Vg))!
    V_q = V_qe[n_g:].tocsr()   # (n_q, N)
    Vg_csc = V_qe[:n_g].tocsc()  # (n_g, N)
    n_q = len(Z_qry)

    J_qg = np.zeros((n_q, n_g), dtype=np.float32)
    for qi in range(n_q):
        s_q, e_q = V_q.indptr[qi], V_q.indptr[qi + 1]
        if e_q == s_q:
            continue
        nz_cols = V_q.indices[s_q:e_q]
        vq_sub = V_q.data[s_q:e_q]
        # Prune negligible tails (< 1e-4) for extra speed with zero accuracy loss
        keep_q = vq_sub > 1e-4
        if not np.any(keep_q):
            continue
        nz_cols = nz_cols[keep_q]
        vq_sub = vq_sub[keep_q]
        sub_csr = Vg_csc[:, nz_cols].tocsr()
        active_g = np.where(np.diff(sub_csr.indptr) > 0)[0]
        if len(active_g) == 0:
            continue
        dense_sub = sub_csr[active_g].toarray()  # (n_act, m)
        mins = np.minimum(dense_sub, vq_sub[None, :]).sum(axis=1)
        J_qg[qi, active_g] = mins / np.maximum(1e-6, 2.0 - mins)

    return J_qg


# ---------------------------------------------------------------------------
# 4. Multi-Hop Personalized PageRank / Manifold Graph Diffusion (Iscen et al.)
# ---------------------------------------------------------------------------

def manifold_graph_diffusion(Z_gal, Z_qry, c_inv, n_c, k_nn=16, alpha=0.82, n_iters=8, gamma_pow=3.0):
    """
    Exact sparse multi-hop Personalized PageRank Diffusion on joint gallery + query graph!
    Computes:
      1) C_diff_cls (n_q, n_c): multi-hop diffused class affinity (bridges cross-view chains & test bursts!)
      2) C_diff_bal (n_q, n_c): class-balanced diffused affinity (protects rare/singleton classes!)
      3) diff_q_mass (n_q,): fraction of diffused neighborhood mass landing on OTHER queries (kills multi-frame unseen!)
      4) top5_q_frac, top10_q_frac: direct top-k query-enrichment ratios!
    """
    Z_all = np.vstack([Z_gal, Z_qry])
    N, n_g, n_q = len(Z_all), len(Z_gal), len(Z_qry)
    S = Z_all @ Z_all.T
    np.fill_diagonal(S, -2.0)

    kk = min(k_nn, N - 1)
    top_idx = np.argpartition(-S, kth=kk, axis=1)[:, :kk]
    top_val = np.take_along_axis(S, top_idx, axis=1)
    w = np.maximum(0.0, top_val) ** gamma_pow

    rows = np.repeat(np.arange(N, dtype=np.int32), kk)
    cols = top_idx.ravel()
    W = sparse.csr_matrix((w.ravel().astype(np.float32), (rows, cols)), shape=(N, N))
    W = 0.5 * (W + W.T)

    deg = np.asarray(W.sum(axis=1)).ravel()
    inv_sqrt_d = np.where(deg > 1e-8, deg ** -0.5, 0.0).astype(np.float32)
    D_inv = sparse.diags(inv_sqrt_d)
    P_sym = D_inv @ W @ D_inv

    # 1) Label Propagation (Standard + Balanced)
    Y_std = np.zeros((N, n_c), dtype=np.float32)
    Y_std[np.arange(n_g), c_inv] = 1.0
    cls_cnt = np.bincount(c_inv, minlength=n_c).astype(np.float32)
    Y_bal = np.zeros((N, n_c), dtype=np.float32)
    Y_bal[np.arange(n_g), c_inv] = 1.0 / np.sqrt(np.maximum(1.0, cls_cnt[c_inv]))

    F_std = Y_std.copy()
    F_bal = Y_bal.copy()
    for _ in range(n_iters):
        F_std = alpha * (P_sym @ F_std) + (1.0 - alpha) * Y_std
        F_bal = alpha * (P_sym @ F_bal) + (1.0 - alpha) * Y_bal

    C_diff_std = F_std[n_g:]
    C_diff_bal = F_bal[n_g:]
    # Row-normalize diffused affinities so scales are invariant across gallery sizes
    C_diff_std = C_diff_std / np.maximum(1e-6, C_diff_std.sum(axis=1, keepdims=True))
    C_diff_bal = C_diff_bal / np.maximum(1e-6, C_diff_bal.sum(axis=1, keepdims=True))

    # 2) Domain-Mass Diffusion (Gallery=0, Query=1, excluding self via row-stochastic P_rw!)
    inv_d = np.where(deg > 1e-8, 1.0 / deg, 0.0).astype(np.float32)
    P_rw = sparse.diags(inv_d) @ W
    y_dom = np.zeros(N, dtype=np.float32)
    y_dom[n_g:] = 1.0
    # Start 1 hop away (P_rw @ y_dom) so self-loop doesn't dominate!
    f_dom = P_rw @ y_dom
    for _ in range(4):
        f_dom = 0.65 * (P_rw @ f_dom) + 0.35 * (P_rw @ y_dom)
    base_q_share = float(np.clip(float(n_q) / float(N), 0.02, 0.98))
    base_q_logit = float(np.log(base_q_share / (1.0 - base_q_share)))

    def _dom_llr(frac_arr):
        fc = np.clip(frac_arr, 0.015, 0.985)
        return np.clip(np.log(fc / (1.0 - fc)) - base_q_logit, -4.5, 4.5).astype(np.float32)

    diff_q_enrich = _dom_llr(f_dom[n_g:])

    # 3) Direct top-5 and top-10 query neighbor log-odds enrichment (invariant to base_q_share shift!)
    top10 = np.argsort(-S[n_g:], axis=1)[:, :10]
    is_q_mat = (top10 >= n_g).astype(np.float32)
    top5_q_enrich = _dom_llr(is_q_mat[:, :5].mean(axis=1))
    top10_q_enrich = _dom_llr(is_q_mat.mean(axis=1))

    return C_diff_std, C_diff_bal, diff_q_enrich, top5_q_enrich, top10_q_enrich


# ---------------------------------------------------------------------------
# 5. Fast Class Aggregations & Peak CSR SIFT Lookup
# ---------------------------------------------------------------------------

def fast_class_stats(S_qg, cols_per_class):
    n_q = S_qg.shape[0]
    n_c = len(cols_per_class)
    C_max = np.empty((n_q, n_c), dtype=np.float32)
    C_top2 = np.empty((n_q, n_c), dtype=np.float32)
    C_mean = np.empty((n_q, n_c), dtype=np.float32)
    best_g = np.empty((n_q, n_c), dtype=np.int32)
    for c, cols in enumerate(cols_per_class):
        sub = S_qg[:, cols]
        am = np.argmax(sub, axis=1)
        mx = sub[np.arange(n_q), am]
        C_max[:, c] = mx
        best_g[:, c] = cols[am]
        C_mean[:, c] = sub.mean(axis=1)
        if len(cols) == 1:
            C_top2[:, c] = mx
        else:
            p2 = np.partition(sub, kth=len(cols) - 2, axis=1)[:, -2:]
            C_top2[:, c] = 0.60 * np.maximum(p2[:, 0], p2[:, 1]) + 0.40 * np.minimum(p2[:, 0], p2[:, 1])
    return C_max, C_top2, C_mean, best_g


def lookup_peak_sift_csr(q_rows, g_rows, c_inv, n_c, n_total, row_ptr, col_idx, feat_vals):
    """
    O(nnz) lookup from symmetric identity-aware peak SIFT CSR cache!
    feat_vals columns: [n_r80, n_r74, n_mut, margin_sum, n_sim_inl, sim_ratio, n_fund_inl, max_kp]
    Also returns max_qq_sim_inl (SIFT RANSAC inliers to OTHER queries in q_rows!).
    """
    g_to_loc = np.full(n_total, -1, dtype=np.int32)
    g_to_loc[g_rows] = np.arange(len(g_rows), dtype=np.int32)
    q_to_loc = np.full(n_total, -1, dtype=np.int32)
    q_to_loc[q_rows] = np.arange(len(q_rows), dtype=np.int32)

    n_q = len(q_rows)
    C_r80 = np.zeros((n_q, n_c), dtype=np.float32)
    C_r74 = np.zeros((n_q, n_c), dtype=np.float32)
    C_mut = np.zeros((n_q, n_c), dtype=np.float32)
    C_marg = np.zeros((n_q, n_c), dtype=np.float32)
    C_sim = np.zeros((n_q, n_c), dtype=np.float32)
    C_srat = np.zeros((n_q, n_c), dtype=np.float32)
    C_fund = np.zeros((n_q, n_c), dtype=np.float32)
    max_qq_sim = np.zeros(n_q, dtype=np.float32)

    for qi, qg in enumerate(q_rows):
        s, e = row_ptr[qg], row_ptr[qg + 1]
        if e == s:
            continue
        nbrs = col_idx[s:e]
        fmat = feat_vals[s:e]
        # Gallery neighbors
        glocs = g_to_loc[nbrs]
        gm = glocs >= 0
        if np.any(gm):
            cls_ids = c_inv[glocs[gm]]
            fg = fmat[gm]
            np.maximum.at(C_r80[qi], cls_ids, fg[:, 0])
            np.maximum.at(C_r74[qi], cls_ids, fg[:, 1])
            np.maximum.at(C_mut[qi], cls_ids, fg[:, 2])
            np.maximum.at(C_marg[qi], cls_ids, fg[:, 3])
            np.maximum.at(C_sim[qi], cls_ids, fg[:, 4])
            np.maximum.at(C_srat[qi], cls_ids, fg[:, 5])
            np.maximum.at(C_fund[qi], cls_ids, fg[:, 6])
        # Query neighbors (for transductive burst verification!)
        qlocs = q_to_loc[nbrs]
        qm = (qlocs >= 0) & (qlocs != qi)
        if np.any(qm):
            max_qq_sim[qi] = float(fmat[qm, 4].max())

    return C_r80, C_r74, C_mut, C_marg, C_sim, C_srat, C_fund, max_qq_sim


def loio_calib(S_gg_m, q_s1):
    loio_max = np.sort(S_gg_m.max(axis=1))
    mu = float(loio_max.mean())
    sd = float(loio_max.std()) + 1e-4
    q90 = float(np.quantile(loio_max, 0.90))
    return ((q_s1 - mu) / sd).astype(np.float32), (q_s1 - q90).astype(np.float32), (np.searchsorted(loio_max, q_s1) / float(len(loio_max))).astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Complete Ground-Up V2 Feature Bundle Builder
# ---------------------------------------------------------------------------

CAND_V2_NAMES = [
    "stage1_score", "diff_stage1",
    "p45_max", "p45_top2", "p45_csls", "diff_p45", "p45_z_loio",
    "p60_max", "p60_top2", "p60_mean", "p60_cent", "p60_csls", "diff_p60", "z_p60_k", "p60_z_loio",
    "p72_max", "diff_p72",
    "raw_max", "diff_raw",
    "subcent_max", "subcent_same_vp", "diff_subcent",
    "lda_max", "lda_top2", "lda_cent", "lda_csls", "diff_lda", "lda_z_loio",
    "wccn_max", "diff_wccn",
    "kiss_max", "diff_kiss",
    "cca_max", "diff_cca",
    "softmax_logp", "diff_softmax_logp",
    "bal_ridge", "diff_bal_ridge",
    "ridge_z", "diff_ridge_z",
    "zhong_jacc", "diff_zhong_jacc",
    "diff_ppr_std", "diff_ppr_bal", "diff_ppr_gap",
    "fv_max", "diff_fv",
    "vl_max", "diff_vl",
    "sift_sim_inl", "sift_fund_inl", "sift_r74", "sift_r80", "sift_mut", "sift_marg", "sift_srat", "diff_sift_sim",
    "vp_has_same", "vp_same_frac", "vp_same_max", "vp_diff_max", "vp_soft_best",
    "qq_p60_max", "diff_qq_p60",
    "cand_log_freq", "cand_hub", "rank_inv"
]  # 67 ultra-rich candidate features!


DEFAULT_STAGE1_CFG = {
    "hyena": {
        "spow": 0.85, "p_mid": 0.58, "reg_mid": 1.8e-3, "d_mid": 640,
        "lda_shrink": 0.22, "k1_zhong": 16, "k2_zhong": 5, "diff_alpha": 0.80,
        "w_p60": 0.24, "w_sub": 0.10, "w_lda": 0.16, "w_wccn": 0.06, "w_kiss": 0.08,
        "w_cca": 0.04, "w_soft": 0.08, "w_zhong": 0.10, "w_ppr": 0.06, "w_fv": 0.04, "w_vl": 0.04,
        "w_sift": 0.018,
    },
    "leopard": {
        "spow": 0.85, "p_mid": 0.60, "reg_mid": 2.0e-3, "d_mid": 768,
        "lda_shrink": 0.22, "k1_zhong": 18, "k2_zhong": 6, "diff_alpha": 0.82,
        "w_p60": 0.24, "w_sub": 0.10, "w_lda": 0.16, "w_wccn": 0.06, "w_kiss": 0.08,
        "w_cca": 0.04, "w_soft": 0.08, "w_zhong": 0.10, "w_ppr": 0.06, "w_fv": 0.04, "w_vl": 0.04,
        "w_sift": 0.020,
    },
}


def build_v2_feature_bundle(
    g_rows, q_rows, species,
    all_identities, all_viewpoints, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
    sift_csr_tuple, top_k_cands=12, stage1_cfg=None, return_stage1_mats=False
):
    cfg = stage1_cfg if stage1_cfg is not None else DEFAULT_STAGE1_CFG[species]
    n_g, n_q = len(g_rows), len(q_rows)
    yg = all_identities[g_rows]
    u_cls, c_inv = np.unique(yg, return_inverse=True)
    n_c = len(u_cls)
    cols_per_class = [np.where(c_inv == c)[0] for c in range(n_c)]
    class_counts = np.array([len(cols) for cols in cols_per_class], dtype=np.float32)
    # Soft-floored log class frequency so n=1 singletons don't suffer an extreme cliff!
    log_class_freq = np.log((np.maximum(1.5, class_counts) + 0.5) / float(n_g))

    # 1. Multi-scale Dual-SVD Whitened MiewID
    proj, eig, _, Xg_n, Xq_n = precompute_pca_basis(miew_all[g_rows], miew_all[q_rows], signed_pow=cfg["spow"])
    d_mid = int(cfg["d_mid"])
    Zg_p45, Zq_p45 = slice_whiten(proj, eig, n_g, power=max(0.30, cfg["p_mid"] - 0.14), reg=cfg["reg_mid"], top_d=int(d_mid * 0.85))
    Zg_p60, Zq_p60 = slice_whiten(proj, eig, n_g, power=cfg["p_mid"], reg=cfg["reg_mid"], top_d=d_mid)
    Zg_p72, Zq_p72 = slice_whiten(proj, eig, n_g, power=min(0.82, cfg["p_mid"] + 0.13), reg=cfg["reg_mid"] * 1.5, top_d=int(d_mid * 1.35))

    mu_raw = np.vstack([Xg_n, Xq_n]).mean(0, keepdims=True)
    Zg_rawc = normalize(Xg_n - mu_raw, axis=1).astype(np.float32)
    Zq_rawc = normalize(Xq_n - mu_raw, axis=1).astype(np.float32)

    # 2. Soft Viewpoint Imputation (using top-256 PCA-whitened components!)
    vp_g, vp_q = all_viewpoints[g_rows], all_viewpoints[q_rows]
    kvp = (vp_g != "MISSING")
    d_vp = min(256, Zg_p60.shape[1])
    vp_clf = LogisticRegression(C=2.0, max_iter=200).fit(Zg_p60[kvp, -d_vp:], vp_g[kvp])
    P_vp_g = vp_clf.predict_proba(Zg_p60[:, -d_vp:]).astype(np.float32)
    P_vp_q = vp_clf.predict_proba(Zq_p60[:, -d_vp:]).astype(np.float32)
    eff_vp_g = np.where(vp_g != "MISSING", vp_g, vp_clf.classes_[np.argmax(P_vp_g, axis=1)])
    eff_vp_q = np.where(vp_q != "MISSING", vp_q, vp_clf.classes_[np.argmax(P_vp_q, axis=1)])

    # 3. Supervised Metric Learning: Shrinkage LDA + WCCN + KISSME + Cross-View CCA
    d_lin = min(512, Zg_p60.shape[1])
    Zg_lin = normalize(Zg_p60[:, -d_lin:], axis=1)
    Zq_lin = normalize(Zq_p60[:, -d_lin:], axis=1)

    valid_m = class_counts[c_inv] >= 2
    n_val_c = int((class_counts >= 2).sum())
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=cfg["lda_shrink"], n_components=min(220, max(2, n_val_c - 1), d_lin))
    lda.fit(Zg_lin[valid_m], yg[valid_m])
    Lg = normalize(lda.transform(Zg_lin), axis=1).astype(np.float32)
    Lq = normalize(lda.transform(Zq_lin), axis=1).astype(np.float32)

    Wg_wccn, Wq_wccn = fit_wccn(Zg_lin, cols_per_class, Zq_lin, reg=0.06, power=0.50)
    Kg_kiss, Kq_kiss = fit_kissme(Zg_lin, cols_per_class, Zq_lin, reg_s=0.14, reg_d=0.14, max_dim=256)
    Cg_cca, Cq_cca = fit_cross_view_cca(Zg_lin, cols_per_class, eff_vp_g, Zq_lin, eff_vp_q, reg=0.18, max_dim=224, n_comp=96)

    # 4. Discriminative Classifiers: Multi-Class Softmax LogReg + Balanced Cosine-Ridge + Standard Ridge
    X_clf_g = np.hstack([Zg_lin[:, -256:], 0.65 * Lg])
    X_clf_q = np.hstack([Zq_lin[:, -256:], 0.65 * Lq])

    # Multi-class Softmax Logistic Regression (fast L-BFGS on ~470 dims!)
    sm_clf = LogisticRegression(C=4.0, max_iter=150, solver="lbfgs")
    sm_clf.fit(X_clf_g, c_inv)
    # Align columns in case any class wasn't present (all 0..n_c-1 are present)
    C_soft_logp = sm_clf.predict_log_proba(X_clf_q).astype(np.float32)

    # Balanced Cosine-Ridge + Standard Ridge
    Y_oh = np.zeros((n_g, n_c), dtype=np.float32)
    Y_oh[np.arange(n_g), c_inv] = 1.0
    Y_bal = np.zeros((n_g, n_c), dtype=np.float32)
    Y_bal[np.arange(n_g), c_inv] = 1.0 / (class_counts[c_inv] ** 0.80)
    A_mat = X_clf_g.T @ X_clf_g + 6.0 * np.eye(X_clf_g.shape[1], dtype=np.float32)
    W_std = np.linalg.solve(A_mat, X_clf_g.T @ Y_oh)
    W_bal = normalize(np.linalg.solve(A_mat, X_clf_g.T @ Y_bal), axis=0)
    C_ridge = X_clf_q @ W_std
    C_ridge_z = (C_ridge - C_ridge.mean(1, keepdims=True)) / (C_ridge.std(1, keepdims=True) + 1e-5)
    C_bal_ridge = (X_clf_q @ W_bal).astype(np.float32)

    # 5. True Zhong k-Reciprocal Re-Ranking + Multi-Hop PPR Manifold Graph Diffusion!
    Zg_mix = normalize(np.hstack([0.75 * Zg_p60, 0.55 * Lg, 0.35 * Kg_kiss]), axis=1)
    Zq_mix = normalize(np.hstack([0.75 * Zq_p60, 0.55 * Lq, 0.35 * Kq_kiss]), axis=1)

    S_zhong = zhong_k_reciprocal_sim(Zg_mix, Zq_mix, k1=int(cfg["k1_zhong"]), k2=int(cfg["k2_zhong"]))
    C_diff_std, C_diff_bal, diff_q_enrich, top5_q_enrich, top10_q_enrich = manifold_graph_diffusion(
        Zg_mix, Zq_mix, c_inv, n_c, k_nn=16, alpha=cfg["diff_alpha"], n_iters=8, gamma_pow=3.0
    )

    # 6. Local Global-Texture Descriptors: RootSIFT Fisher Vectors + VLAD
    pfv, efv, _, _, _ = precompute_pca_basis(fv_all[g_rows], fv_all[q_rows], signed_pow=1.0)
    Fg_fv, Fq_fv = slice_whiten(pfv, efv, n_g, power=0.40, reg=1e-2, top_d=512)
    pvl, evl, _, _, _ = precompute_pca_basis(vlad_all[g_rows], vlad_all[q_rows], signed_pow=1.0)
    Vg_vl, Vq_vl = slice_whiten(pvl, evl, n_g, power=0.40, reg=1e-2, top_d=512)

    # 7. Pairwise Similarities & Class Aggregations
    S_p45 = Zq_p45 @ Zg_p45.T
    S_p60 = Zq_p60 @ Zg_p60.T
    S_p72 = Zq_p72 @ Zg_p72.T
    S_raw = Zq_rawc @ Zg_rawc.T
    S_lda = Lq @ Lg.T
    S_wccn = Wq_wccn @ Wg_wccn.T
    S_kiss = Kq_kiss @ Kg_kiss.T
    S_cca = Cq_cca @ Cg_cca.T
    S_fv = Fq_fv @ Fg_fv.T
    S_vl = Vq_vl @ Vg_vl.T

    Cp45_max, Cp45_t2, _, _ = fast_class_stats(S_p45, cols_per_class)
    Cp60_max, Cp60_t2, Cp60_mean, best_g_p60 = fast_class_stats(S_p60, cols_per_class)
    Cp72_max, _, _, _ = fast_class_stats(S_p72, cols_per_class)
    Craw_max, _, _, _ = fast_class_stats(S_raw, cols_per_class)
    Clda_max, Clda_t2, _, _ = fast_class_stats(S_lda, cols_per_class)
    Cwccn_max, _, _, _ = fast_class_stats(S_wccn, cols_per_class)
    Ckiss_max, _, _, _ = fast_class_stats(S_kiss, cols_per_class)
    Ccca_max, _, _, _ = fast_class_stats(S_cca, cols_per_class)
    Czhong_max, _, _, _ = fast_class_stats(S_zhong, cols_per_class)
    Cfv_max, _, _, _ = fast_class_stats(S_fv, cols_per_class)
    Cvl_max, _, _, _ = fast_class_stats(S_vl, cols_per_class)

    # 8. Viewpoint-Conditioned Sub-Center Prototypes (denoises within-flank without mixing left & right!)
    sub_vecs, sub_cls, sub_vps = [], [], []
    for c, cols in enumerate(cols_per_class):
        vps_c = eff_vp_g[cols]
        for v_str in np.unique(vps_c):
            vc_idx = cols[vps_c == v_str]
            sub_vecs.append(Zg_p60[vc_idx].mean(axis=0))
            sub_cls.append(c)
            sub_vps.append(v_str)
    sub_vecs = normalize(np.vstack(sub_vecs), axis=1).astype(np.float32)
    sub_cls = np.array(sub_cls, dtype=np.int32)
    sub_vps = np.array(sub_vps, dtype=object)
    S_q_sub = Zq_p60 @ sub_vecs.T  # (n_q, n_sub)

    C_sub_max = np.full((n_q, n_c), -1.0, dtype=np.float32)
    C_sub_same = np.full((n_q, n_c), -0.15, dtype=np.float32)
    for sc_idx in range(len(sub_cls)):
        c = sub_cls[sc_idx]
        v_str = sub_vps[sc_idx]
        col_sim = S_q_sub[:, sc_idx]
        C_sub_max[:, c] = np.maximum(C_sub_max[:, c], col_sim)
        sm = (eff_vp_q == v_str)
        if np.any(sm):
            C_sub_same[sm, c] = np.maximum(C_sub_same[sm, c], col_sim[sm])

    # Single-class centroids in p60 and LDA
    cent_p60 = normalize(np.vstack([Zg_p60[cols].mean(0) for cols in cols_per_class]), axis=1).astype(np.float32)
    cent_lda = normalize(np.vstack([Lg[cols].mean(0) for cols in cols_per_class]), axis=1).astype(np.float32)
    C_cent_p60 = Zq_p60 @ cent_p60.T
    C_cent_lda = Lq @ cent_lda.T

    # 9. Peak SIFT (Symmetric Scale-Bounded 4-DOF Sim + Epipolar Fundamental Matrix RANSAC)
    row_ptr, col_idx, feat_vals = sift_csr_tuple
    Cs_r80, Cs_r74, Cs_mut, Cs_marg, Cs_sim, Cs_srat, Cs_fund, max_qq_sift = lookup_peak_sift_csr(
        q_rows, g_rows, c_inv, n_c, len(miew_all), row_ptr, col_idx, feat_vals
    )
    sift_term = (
        0.55 * np.log1p(Cs_sim) + 0.35 * np.log1p(Cs_fund) + 0.30 * np.log1p(Cs_r74) + 1.0 * ((Cs_sim >= 8) | (Cs_fund >= 10))
    ).astype(np.float32)
    sift_bonus = cfg["w_sift"] * sift_term

    # 10. Gallery LOIO Calibration & Impostor Hubness
    same_id_gg = (c_inv[:, None] == c_inv[None, :])
    S_gg_p45 = Zg_p45 @ Zg_p45.T; S_gg_p45[same_id_gg] = -2.0
    S_gg_p60 = Zg_p60 @ Zg_p60.T; S_gg_p60[same_id_gg] = -2.0
    S_gg_raw = Zg_rawc @ Zg_rawc.T; S_gg_raw[same_id_gg] = -2.0
    S_gg_lda = Lg @ Lg.T; S_gg_lda[same_id_gg] = -2.0
    S_gg_kiss = Kg_kiss @ Kg_kiss.T; S_gg_kiss[same_id_gg] = -2.0

    mu45, sd45 = float(S_gg_p45.max(1).mean()), float(S_gg_p45.max(1).std()) + 1e-4
    mu60, sd60 = float(S_gg_p60.max(1).mean()), float(S_gg_p60.max(1).std()) + 1e-4
    mulda, sdlda = float(S_gg_lda.max(1).mean()), float(S_gg_lda.max(1).std()) + 1e-4

    gal_hub_img = np.partition(S_gg_p60, kth=max(0, n_g - 16), axis=1)[:, -15:].mean(axis=1)
    class_hub_p60 = np.array([gal_hub_img[cols].max() for cols in cols_per_class], dtype=np.float32)

    def id_csls(M):
        bg = -np.sort(-M, axis=1)[:, 1:min(16, n_c)].mean(axis=1, keepdims=True)
        return M - 0.50 * bg

    Cp45_csls = id_csls(Cp45_max)
    Cp60_csls = id_csls(Cp60_max) - 0.30 * class_hub_p60[None, :]
    Clda_csls = id_csls(Clda_max)

    # 11. Query-to-Query Encounter Graph Clustering (Multi-hop Connected Components!)
    S_qq_mix = Zq_mix @ Zq_mix.T
    S_qq_raw = Zq_rawc @ Zq_rawc.T
    np.fill_diagonal(S_qq_mix, -2.0)
    np.fill_diagonal(S_qq_raw, -2.0)
    max_qq_mix = S_qq_mix.max(axis=1)
    max_qq_raw = S_qq_raw.max(axis=1)

    W_qq = np.where((S_qq_mix >= 0.36) | (S_qq_raw >= 0.68), np.maximum(0.0, S_qq_mix) ** 3.5, 0.0)
    w_qq_sum = W_qq.sum(axis=1, keepdims=True)
    has_qq = (w_qq_sum[:, 0] > 0)
    C_qq_p60 = Cp60_max.copy()
    if np.any(has_qq):
        C_qq_p60[has_qq] = 0.65 * Cp60_max[has_qq] + 0.35 * ((W_qq[has_qq] @ Cp60_max) / w_qq_sum[has_qq])

    # Connected-component test clusters for cluster-level open-set inference!
    nc_q, q_comp = connected_components((S_qq_mix >= 0.38) | (S_qq_raw >= 0.72), directed=False)

    # Standardize softmax logp row-wise so it scales cleanly alongside cosine scores
    C_soft_z = (C_soft_logp - C_soft_logp.mean(1, keepdims=True)) / (C_soft_logp.std(1, keepdims=True) + 1e-4)

    # 12. Stage-1 Master Retrieval Fusion (selects Top-K candidates!)
    m_sub_csls = id_csls(C_sub_max)
    m_wccn_csls = id_csls(Cwccn_max)
    m_kiss_csls = id_csls(Ckiss_max)
    m_cca_csls = id_csls(Ccca_max)
    m_soft_sc = (0.10 * C_soft_z).astype(np.float32)
    m_ppr_sc = (2.5 * C_diff_bal).astype(np.float32)
    m_fv_csls = id_csls(Cfv_max)
    m_vl_csls = id_csls(Cvl_max)

    if return_stage1_mats:
        return {
            "u_cls": u_cls,
            "mats": {
                "w_p60": Cp60_csls, "w_sub": m_sub_csls, "w_lda": Clda_csls,
                "w_wccn": m_wccn_csls, "w_kiss": m_kiss_csls, "w_cca": m_cca_csls,
                "w_soft": m_soft_sc, "w_zhong": Czhong_max, "w_ppr": m_ppr_sc,
                "w_fv": m_fv_csls, "w_vl": m_vl_csls, "w_sift": sift_term,
            },
        }

    C_stage1 = (
        cfg["w_p60"] * Cp60_csls
        + cfg["w_sub"] * m_sub_csls
        + cfg["w_lda"] * Clda_csls
        + cfg["w_wccn"] * m_wccn_csls
        + cfg["w_kiss"] * m_kiss_csls
        + cfg["w_cca"] * m_cca_csls
        + cfg["w_soft"] * m_soft_sc
        + cfg["w_zhong"] * Czhong_max
        + cfg["w_ppr"] * m_ppr_sc
        + cfg["w_fv"] * m_fv_csls
        + cfg["w_vl"] * m_vl_csls
        + sift_bonus
    )

    K = min(top_k_cands, n_c)
    top_cands = np.argsort(-C_stage1, axis=1)[:, :K]
    r2d = np.arange(n_q)[:, None]

    def gather(M):
        return M[r2d, top_cands]

    def diff_best(V):
        i2 = np.argsort(-V, axis=1)[:, :2]
        v1 = V[np.arange(n_q), i2[:, 0]]
        v2 = V[np.arange(n_q), i2[:, 1]]
        return V - np.where(np.arange(K)[None, :] == i2[:, :1], v2[:, None], v1[:, None])

    # Viewpoint compatibility & same-VP vs diff-VP max sim for top-K candidates
    vp_same_max = np.full((n_q, K), -0.15, dtype=np.float32)
    vp_diff_max = np.full((n_q, K), -0.15, dtype=np.float32)
    vp_same_frac = np.zeros((n_q, K), dtype=np.float32)
    vp_has_same = np.zeros((n_q, K), dtype=np.float32)
    vp_soft_best = np.sum(P_vp_q[:, None, :] * P_vp_g[gather(best_g_p60)], axis=2)

    for qi in range(n_q):
        vq_str = eff_vp_q[qi]
        s_row = S_p60[qi]
        for ki in range(K):
            cid = top_cands[qi, ki]
            cols = cols_per_class[cid]
            sm = (eff_vp_g[cols] == vq_str)
            ns = int(sm.sum())
            if ns > 0:
                vp_has_same[qi, ki] = 1.0
                vp_same_frac[qi, ki] = float(ns) / len(cols)
                vp_same_max[qi, ki] = s_row[cols[sm]].max()
            if ns < len(cols):
                vp_diff_max[qi, ki] = s_row[cols[~sm]].max()

    g_st1 = gather(C_stage1)
    g_p45 = gather(Cp45_max)
    g_p60 = gather(Cp60_max)
    g_p72 = gather(Cp72_max)
    g_raw = gather(Craw_max)
    g_sub = gather(C_sub_max)
    g_lda = gather(Clda_max)
    g_wccn = gather(Cwccn_max)
    g_kiss = gather(Ckiss_max)
    g_cca = gather(Ccca_max)
    g_soft = gather(C_soft_logp)
    g_brid = gather(C_bal_ridge)
    g_ridz = gather(C_ridge_z)
    g_zh = gather(Czhong_max)
    g_dstd = gather(C_diff_std)
    g_dbal = gather(C_diff_bal)
    g_fv = gather(Cfv_max)
    g_vl = gather(Cvl_max)
    g_sim = np.log1p(gather(Cs_sim))
    g_fnd = np.log1p(gather(Cs_fund))
    g_r74 = np.log1p(gather(Cs_r74))
    g_r80 = np.log1p(gather(Cs_r80))
    g_mut = np.log1p(gather(Cs_mut))
    g_qq = gather(C_qq_p60)

    z_p60_k = (g_p60 - g_p60.mean(1, keepdims=True)) / (g_p60.std(1, keepdims=True) + 1e-4)
    cand_log_freq = log_class_freq[top_cands]
    cand_hub = class_hub_p60[top_cands]
    rank_inv = (1.0 / (1.0 + np.arange(K, dtype=np.float32)))[None, :].repeat(n_q, axis=0)

    X_cand = np.stack([
        g_st1, diff_best(g_st1),
        g_p45, gather(Cp45_t2), gather(Cp45_csls), diff_best(g_p45), (g_p45 - mu45) / sd45,
        g_p60, gather(Cp60_t2), gather(Cp60_mean), gather(C_cent_p60), gather(Cp60_csls), diff_best(g_p60), z_p60_k, (g_p60 - mu60) / sd60,
        g_p72, diff_best(g_p72),
        g_raw, diff_best(g_raw),
        g_sub, gather(C_sub_same), diff_best(g_sub),
        g_lda, gather(Clda_t2), gather(C_cent_lda), gather(Clda_csls), diff_best(g_lda), (g_lda - mulda) / sdlda,
        g_wccn, diff_best(g_wccn),
        g_kiss, diff_best(g_kiss),
        g_cca, diff_best(g_cca),
        g_soft, diff_best(g_soft),
        g_brid, diff_best(g_brid),
        g_ridz, diff_best(g_ridz),
        g_zh, diff_best(g_zh),
        g_dstd, g_dbal, diff_best(g_dbal),
        g_fv, diff_best(g_fv),
        g_vl, diff_best(g_vl),
        g_sim, g_fnd, g_r74, g_r80, g_mut, gather(Cs_marg), gather(Cs_srat), diff_best(g_sim),
        vp_has_same, vp_same_frac, vp_same_max, vp_diff_max, vp_soft_best,
        g_qq, diff_best(g_qq),
        cand_log_freq, cand_hub, rank_inv,
    ], axis=2).astype(np.float32)  # (n_q, K, 67)

    # 13. Query-Level Open-Set Features (including Graph-Community Enrichment & Multi-Frame Cluster Evidence!)
    def top_prof(M):
        s = -np.sort(-M, axis=1)[:, :10]
        bg = s[:, 1:10].mean(1)
        sd = s[:, 1:10].std(1) + 1e-4
        return [s[:, 0], s[:, 1], s[:, 0] - s[:, 1], s[:, 0] - s[:, 3], s[:, 0] - s[:, 5], bg, (s[:, 0] - bg) / sd]

    p45_s1 = Cp45_max.max(axis=1)
    p60_s1 = Cp60_max.max(axis=1)
    raw_s1 = Craw_max.max(axis=1)
    lda_s1 = Clda_max.max(axis=1)
    kiss_s1 = Ckiss_max.max(axis=1)

    z45, dq45, ec45 = loio_calib(S_gg_p45, p45_s1)
    z60, dq60, ec60 = loio_calib(S_gg_p60, p60_s1)
    zraw, dqraw, ecraw = loio_calib(S_gg_raw, raw_s1)
    zlda, dqlda, eclda = loio_calib(S_gg_lda, lda_s1)
    zkiss, dqkiss, eckiss = loio_calib(S_gg_kiss, kiss_s1)

    # Cluster-level aggregates across multi-hop connected components of queries!
    clust_size = np.ones(n_q, dtype=np.float32)
    clust_max_z60 = z60.copy()
    clust_mean_z60 = z60.copy()
    clust_max_sim_inl = np.log1p(Cs_sim.max(axis=1))
    clust_mean_q_enrich = diff_q_enrich.copy()

    for cid in range(nc_q):
        m_c = (q_comp == cid)
        sz = int(m_c.sum())
        if sz >= 2:
            clust_size[m_c] = float(sz)
            clust_max_z60[m_c] = z60[m_c].max()
            clust_mean_z60[m_c] = z60[m_c].mean()
            clust_max_sim_inl[m_c] = clust_max_sim_inl[m_c].max()
            clust_mean_q_enrich[m_c] = diff_q_enrich[m_c].mean()

    sm_probs = np.exp(C_soft_logp)
    sm_ent = -np.sum(sm_probs * C_soft_logp, axis=1)

    open_cols = (
        top_prof(C_stage1)
        + top_prof(Cp60_max)
        + top_prof(Cp45_max)[:4]
        + top_prof(Craw_max)[:5]
        + top_prof(C_sub_max)[:5]
        + top_prof(Clda_max)[:5]
        + top_prof(Ckiss_max)[:4]
        + top_prof(Czhong_max)[:4]
        + top_prof(C_diff_bal)[:4]
        + top_prof(Cfv_max)[:3]
        + top_prof(Cvl_max)[:3]
        + [
            z45, dq45, ec45,
            z60, dq60, ec60,
            zraw, dqraw, ecraw,
            zlda, dqlda, eclda,
            zkiss, dqkiss, eckiss,
            C_soft_logp.max(1), sm_ent,
            C_bal_ridge.max(1),
            C_ridge_z.max(1),
            np.log1p(Cs_sim.max(1)),
            np.log1p(Cs_fund.max(1)),
            np.log1p(Cs_r74.max(1)),
            g_sim[:, 0], g_fnd[:, 0], g_r74[:, 0],
            vp_has_same[:, 0], vp_same_frac[:, 0], vp_same_max[:, 0], vp_diff_max[:, 0], vp_soft_best[:, 0],
            cand_log_freq[:, 0], cand_hub[:, 0],
            np.log1p(kp_counts[q_rows].astype(np.float32)),
            miew_norms[q_rows].astype(np.float32),
            # Graph-community open-set features (scale-invariant log-odds & query-vs-gallery gaps!)
            diff_q_enrich, top5_q_enrich, top10_q_enrich,
            max_qq_mix, max_qq_raw, max_qq_mix - p60_s1, max_qq_raw - raw_s1, np.log1p(max_qq_sift),
            np.log1p(clust_size),
            clust_max_z60, clust_mean_z60, z60 - clust_max_z60,
            clust_max_sim_inl, clust_mean_q_enrich,
        ]
    )
    X_open_base = np.column_stack(open_cols).astype(np.float32)

    return {
        "u_cls": u_cls,
        "top_cands": top_cands,
        "cand_names": u_cls[top_cands],
        "stage1_top_scores": g_st1,
        "raw_s1": raw_s1,
        "max_sim_inl": Cs_sim.max(axis=1),
        "X_cand": X_cand,              # (n_q, K, 67)
        "X_open_base": X_open_base,    # (n_q, 97)
        "W_qq": W_qq,
        "w_qq_sum": w_qq_sum,
        "has_qq": has_qq,
        "q_comp": q_comp,
        "nc_q": nc_q,
    }
