import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import normalize

from cv_framework import build_or_load_cv, evaluate_cv_predictions, EST_PI_NEW, DATA_DIR


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


def aggregate_class_scores(S_qg, gal_ids, unique_classes=None, class_inv=None, mode="max", topk=2, tau=18.0, size_pen=0.0):
    """
    S_qg: (n_q, n_g) pairwise similarity matrix (masked entries set to -1e9).
    Returns:
      C_scores: (n_q, n_classes) score per unique gallery identity
      unique_classes: array of class strings
    """
    if unique_classes is None:
        unique_classes, class_inv = np.unique(gal_ids, return_inverse=True)
    n_q = S_qg.shape[0]
    n_c = len(unique_classes)
    C_scores = np.full((n_q, n_c), -1e9, dtype=np.float64)

    for c in range(n_c):
        cols = np.where(class_inv == c)[0]
        sub = S_qg[:, cols]  # (n_q, m_c)
        if mode == "max" or sub.shape[1] == 1:
            sc = sub.max(axis=1)
        elif mode == "top2_mean":
            part = np.sort(sub, axis=1)[:, -min(topk, sub.shape[1]):]
            valid = part > -1e5
            sc = np.where(valid.any(axis=1), np.where(valid, part, 0.0).sum(axis=1) / np.maximum(1, valid.sum(axis=1)), -1e9)
        elif mode == "lse":
            # Blend max + log-sum-exp over valid gallery sightings
            mx = sub.max(axis=1, keepdims=True)
            valid = (sub > -1e5)
            m_eff = np.maximum(1.0, valid.sum(axis=1))
            exps = np.where(valid, np.exp(np.clip(tau * (sub - mx), -50, 0)), 0.0)
            sc = mx[:, 0] + (np.log(np.maximum(1e-12, exps.sum(axis=1))) - size_pen * np.log(m_eff)) / tau
        elif mode == "max_plus_mean":
            mx = sub.max(axis=1)
            valid = sub > -1e5
            mn = np.where(valid.any(axis=1), np.where(valid, sub, 0.0).sum(axis=1) / np.maximum(1, valid.sum(axis=1)), -1e9)
            sc = 0.80 * mx + 0.20 * mn
        C_scores[:, c] = sc
    return C_scores, unique_classes


def apply_csls(S_qg, S_gg_clean, S_qq_or_qg, k=10):
    """Cross-domain similarity local scaling (hubness reduction)."""
    # Gallery neighborhood mean sim
    k_g = min(k, S_gg_clean.shape[1] - 1)
    r_g = np.sort(S_gg_clean, axis=1)[:, -k_g:].mean(axis=1)  # (n_g,)
    # Query neighborhood mean sim against gallery
    k_q = min(k, S_qg.shape[1])
    r_q = np.sort(S_qg, axis=1)[:, -k_q:].mean(axis=1)  # (n_q,)
    return 2.0 * S_qg - r_q[:, None] - r_g[None, :]


def fit_gallery_open_set_calibrator(S_gg, gal_ids, gal_bursts, pi_new, agg_kwargs):
    """
    Zero-leakage calibrator trained purely inside gallery via LOBO (known) vs LOIO (new_individual)!
    """
    unique_classes, class_inv = np.unique(gal_ids, return_inverse=True)
    n_g = len(gal_ids)

    # 1) LOIO (unseen individual simulation): mask all gallery items with same identity
    S_loio = S_gg.copy()
    same_id = (gal_ids[:, None] == gal_ids[None, :])
    S_loio[same_id] = -1e9
    C_new, _ = aggregate_class_scores(S_loio, gal_ids, unique_classes, class_inv, **agg_kwargs)

    # 2) LOBO (known individual simulation): mask same burst, keep rows whose identity still has >=1 other burst
    S_lobo = S_gg.copy()
    same_burst = (gal_bursts[:, None] == gal_bursts[None, :])
    S_lobo[same_burst] = -1e9
    # Check which rows still have at least one valid gallery item of their own identity
    has_other_burst = ((same_id & (~same_burst)).sum(axis=1) > 0)
    lobo_rows = np.where(has_other_burst)[0]
    C_known, _ = aggregate_class_scores(S_lobo[lobo_rows], gal_ids, unique_classes, class_inv, **agg_kwargs)
    true_class_idx = class_inv[lobo_rows]

    # Fit temperature tau on top-15 candidate scores for known conditional probabilities
    topM = min(15, len(unique_classes))
    top_idx = np.argsort(-C_known, axis=1)[:, :topM]
    top_sc = np.take_along_axis(C_known, top_idx, axis=1)
    # Find where true_class_idx is in top_idx
    hit_mask = (top_idx == true_class_idx[:, None])
    has_hit = hit_mask.any(axis=1)
    hit_pos = hit_mask[has_hit].argmax(axis=1)
    top_sc_hit = top_sc[has_hit]

    def nll_tau(tau):
        logits = tau * (top_sc_hit - top_sc_hit[:, :1])
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        return -np.mean(log_probs[np.arange(len(hit_pos)), hit_pos])

    res_tau = minimize_scalar(nll_tau, bounds=(2.0, 80.0), method="bounded")
    tau_opt = float(res_tau.x)

    # Extract open-set features from sorted class score profile
    def score_features(C_mat):
        s_sorted = -np.sort(-C_mat, axis=1)[:, :10]
        s1 = s_sorted[:, 0]
        s2 = s_sorted[:, 1]
        s3 = s_sorted[:, 2]
        s5 = s_sorted[:, 4]
        mean5 = s_sorted[:, :5].mean(axis=1)
        std5 = s_sorted[:, :5].std(axis=1)
        # Entropy of top-10 softmax
        probs = softmax(tau_opt * s_sorted, axis=1)
        ent = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)), axis=1)
        return np.column_stack([s1, s2, s1 - s2, s1 - s3, s1 - s5, s1 / ( np.abs(mean5) + 1e-4 ), std5, ent, probs[:, 0]])

    X_cal_known = score_features(C_known)
    X_cal_new = score_features(C_new)
    X_cal = np.vstack([X_cal_known, X_cal_new])
    y_cal = np.array([0] * len(X_cal_known) + [1] * len(X_cal_new))
    w_known = (1.0 - pi_new) / len(X_cal_known)
    w_new = pi_new / len(X_cal_new)
    sw = np.array([w_known] * len(X_cal_known) + [w_new] * len(X_cal_new)) * (len(X_cal))

    clf_new = LogisticRegression(C=2.0, max_iter=500)
    clf_new.fit(X_cal, y_cal, sample_weight=sw)
    return tau_opt, clf_new, score_features, unique_classes, class_inv


def decode_predictions(C_val, unique_classes, tau_opt, clf_new, feat_fn, decision_mode="bayes_opt"):
    topM = min(15, len(unique_classes))
    top_idx = np.argsort(-C_val, axis=1)[:, :topM]
    top_sc = np.take_along_axis(C_val, top_idx, axis=1)
    cond_probs = softmax(tau_opt * top_sc, axis=1)  # (n_q, topM)
    p_new = clf_new.predict_proba(feat_fn(C_val))[:, 1]  # (n_q,)

    preds = []
    for i in range(len(C_val)):
        cands = [unique_classes[j] for j in top_idx[i, :5]]
        if decision_mode == "hard_thresh":
            if p_new[i] >= 0.5:
                preds.append(["new_individual"] + cands[:4])
            else:
                preds.append(cands[:5])
        elif decision_mode == "thresh_else_rank2":
            if p_new[i] >= 0.5:
                preds.append(["new_individual"] + cands[:4])
            else:
                preds.append([cands[0], "new_individual"] + cands[1:4])
        else:  # Exact Bayes-Optimal Expected MAP@5 Posterior Ranking!
            known_post = (1.0 - p_new[i]) * cond_probs[i, :5]
            items = [("new_individual", p_new[i])] + [(cands[r], known_post[r]) for r in range(5)]
            items.sort(key=lambda x: -x[1])
            preds.append([lbl for lbl, _ in items[:5]])
    return preds


# --- Feature & Metric Learning Transforms (fitted per fold, per species!) ---

def transform_raw_l2(Xg, yg, Xq, **kw):
    return normalize(Xg, axis=1), normalize(Xq, axis=1)

def transform_center_l2_inductive(Xg, yg, Xq, **kw):
    Xg_n = normalize(Xg, axis=1)
    Xq_n = normalize(Xq, axis=1)
    mu = Xg_n.mean(axis=0, keepdims=True)
    return normalize(Xg_n - mu, axis=1), normalize(Xq_n - mu, axis=1)

def transform_center_l2_transductive(Xg, yg, Xq, **kw):
    Xg_n = normalize(Xg, axis=1)
    Xq_n = normalize(Xq, axis=1)
    mu = np.vstack([Xg_n, Xq_n]).mean(axis=0, keepdims=True)
    return normalize(Xg_n - mu, axis=1), normalize(Xq_n - mu, axis=1)

def transform_power_center_l2(Xg, yg, Xq, beta=0.75, **kw):
    Xg_p = np.sign(Xg) * (np.abs(Xg) ** beta)
    Xq_p = np.sign(Xq) * (np.abs(Xq) ** beta)
    Xg_n = normalize(Xg_p, axis=1)
    Xq_n = normalize(Xq_p, axis=1)
    mu = np.vstack([Xg_n, Xq_n]).mean(axis=0, keepdims=True)
    return normalize(Xg_n - mu, axis=1), normalize(Xq_n - mu, axis=1)

def transform_pca_whiten(Xg, yg, Xq, n_comp=512, power=0.35, transductive=True, **kw):
    Xg_c, Xq_c = transform_center_l2_transductive(Xg, yg, Xq) if transductive else transform_center_l2_inductive(Xg, yg, Xq)
    X_fit = np.vstack([Xg_c, Xq_c]) if transductive else Xg_c
    d = min(n_comp, X_fit.shape[0] - 1, X_fit.shape[1])
    pca = PCA(n_components=d, random_state=42)
    pca.fit(X_fit)
    scale = (pca.explained_variance_ + 1e-5) ** (-0.5 * power)
    Zg = pca.transform(Xg_c) * scale
    Zq = pca.transform(Xq_c) * scale
    return normalize(Zg, axis=1), normalize(Zq, axis=1)

def transform_wccn(Xg, yg, Xq, reg=0.10, power=0.50, **kw):
    """
    Within-Class Covariance Normalization (Bishop PRML / Speaker & ReID metric learning):
    Suppresses viewpoint (left-vs-right), pose, and lighting variation within multi-image identities!
    """
    Xg_c, Xq_c = transform_center_l2_transductive(Xg, yg, Xq)
    # Compute class residuals x - mu_c for classes with >= 2 samples
    residuals = []
    for ident in np.unique(yg):
        idx = np.where(yg == ident)[0]
        if len(idx) >= 2:
            sub = Xg_c[idx]
            residuals.append(sub - sub.mean(axis=0, keepdims=True))
    R = np.vstack(residuals)  # (N_res, D)
    # Fast SVD of R / sqrt(N_res): Sw = (R^T R)/N_res = V S^2 V^T
    # (Sw + reg*I)^(-power) = I*reg^(-power) + V ( (S^2/N + reg)^(-power) - reg^(-power) ) V^T
    _, s_vals, Vt = np.linalg.svd(R / np.sqrt(len(R)), full_matrices=False)
    eig = s_vals ** 2
    base = (reg) ** (-power)
    diff = ((eig + reg) ** (-power)) - base

    def apply_w(X):
        proj = X @ Vt.T  # (N, r)
        return base * X + (proj * diff) @ Vt

    return normalize(apply_w(Xg_c), axis=1), normalize(apply_w(Xq_c), axis=1)

def transform_lda_blend(Xg, yg, Xq, shrinkage=0.25, w_lda=0.55, **kw):
    """
    Blend WCCN/Centered embedding with Shrinkage-LDA Fisher discriminant subspace!
    """
    Xg_w, Xq_w = transform_wccn(Xg, yg, Xq, reg=0.08, power=0.5)
    # Keep classes with >= 2 samples for LDA fitting
    vc = pd.Series(yg).value_counts()
    valid_ids = set(vc[vc >= 2].index)
    mask = np.array([y in valid_ids for y in yg])
    # First PCA to 600 dims on gallery+query for well-conditioned fast Shrinkage LDA
    X_all = np.vstack([Xg_w, Xq_w])
    d_pca = min(600, X_all.shape[0] - 2)
    pca = PCA(n_components=d_pca, random_state=42).fit(X_all)
    Pg = pca.transform(Xg_w)
    Pq = pca.transform(Xq_w)

    n_cls = len(valid_ids)
    n_lda = min(200, n_cls - 1, d_pca)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrinkage, n_components=n_lda)
    lda.fit(Pg[mask], yg[mask])
    Lg = normalize(lda.transform(Pg), axis=1)
    Lq = normalize(lda.transform(Pq), axis=1)
    # Concatenate scaled cosine spaces: s_comb(x, y) = (1-w)*s_wccn + w*s_lda
    Zg = np.hstack([np.sqrt(1.0 - w_lda) * Xg_w, np.sqrt(w_lda) * Lg])
    Zq = np.hstack([np.sqrt(1.0 - w_lda) * Xq_w, np.sqrt(w_lda) * Lq])
    return Zg, Zq

def apply_alpha_qe(Zg, Zq, top_k=4, alpha=3.0, transductive=True):
    """Alpha Query Expansion (Chum et al. / Radenovic et al.)."""
    if transductive:
        Z_all = np.vstack([Zg, Zq])
        S = Z_all @ Z_all.T
        idx = np.argsort(-S, axis=1)[:, :top_k]
        Z_new = np.zeros_like(Z_all)
        for i in range(len(Z_all)):
            nbrs = idx[i]
            sims = np.maximum(0.0, S[i, nbrs]) ** alpha
            Z_new[i] = Z_all[i] + np.sum(sims[:, None] * Z_all[nbrs], axis=0)
        Z_new = normalize(Z_new, axis=1)
        return Z_new[:len(Zg)], Z_new[len(Zg):]
    else:
        # Inductive QE: expand gallery within gallery, expand query using gallery only
        Sg = Zg @ Zg.T
        idx_g = np.argsort(-Sg, axis=1)[:, :top_k]
        Zg_new = np.zeros_like(Zg)
        for i in range(len(Zg)):
            nbrs = idx_g[i]
            sims = np.maximum(0.0, Sg[i, nbrs]) ** alpha
            Zg_new[i] = Zg[i] + np.sum(sims[:, None] * Zg[nbrs], axis=0)
        Zg_new = normalize(Zg_new, axis=1)

        Sqg = Zq @ Zg_new.T
        idx_q = np.argsort(-Sqg, axis=1)[:, :max(1, top_k - 1)]
        Zq_new = np.zeros_like(Zq)
        for i in range(len(Zq)):
            nbrs = idx_q[i]
            sims = np.maximum(0.0, Sqg[i, nbrs]) ** alpha
            Zq_new[i] = Zq[i] + 0.5 * np.sum(sims[:, None] * Zg_new[nbrs], axis=0)
        Zq_new = normalize(Zq_new, axis=1)
        return Zg_new, Zq_new


def run_experiment(name, cv, train_df, X_all, transform_fn, decision_mode="bayes_opt",
                   agg_kwargs=None, use_csls=False, qe_mode=None, **tf_kwargs):
    if agg_kwargs is None:
        agg_kwargs = {"mode": "max"}
    t0 = time.time()
    X_norm_base = normalize(X_all, axis=1)
    fold_preds = {}

    for fold in cv["folds"]:
        k = fold["fold"]
        g_idx = fold["gallery_idx"]
        v_idx = fold["val_idx"]
        preds_fold = [None] * len(v_idx)

        for sp in ["hyena", "leopard"]:
            g_sp_mask = (train_df["species"].values[g_idx] == sp)
            v_sp_mask = (train_df["species"].values[v_idx] == sp)
            g_rows = g_idx[g_sp_mask]
            v_rows = v_idx[v_sp_mask]
            v_local_pos = np.where(v_sp_mask)[0]

            Xg_raw = X_all[g_rows]
            Xq_raw = X_all[v_rows]
            yg = train_df["identity"].values[g_rows]
            gal_bursts = compute_burst_ids(X_norm_base[g_rows], yg, thresh=0.875)

            Zg, Zq = transform_fn(Xg_raw, yg, Xq_raw, **tf_kwargs)
            if qe_mode == "inductive":
                Zg, Zq = apply_alpha_qe(Zg, Zq, top_k=3, alpha=3.0, transductive=False)
            elif qe_mode == "transductive":
                Zg, Zq = apply_alpha_qe(Zg, Zq, top_k=4, alpha=3.5, transductive=True)

            S_gg = Zg @ Zg.T
            S_qg = Zq @ Zg.T

            if use_csls:
                S_gg_nodiag = S_gg.copy()
                np.fill_diagonal(S_gg_nodiag, -1e9)
                # Mask same identity in CSLS hubness radius so large classes aren't penalized!
                S_gg_imp = S_gg.copy()
                S_gg_imp[yg[:, None] == yg[None, :]] = -1e9
                S_qg = apply_csls(S_qg, S_gg_imp, S_qg, k=15)
                S_gg = apply_csls(S_gg, S_gg_imp, S_gg_imp, k=15)

            np.fill_diagonal(S_gg, -1e9)
            tau_opt, clf_new, feat_fn, u_cls, c_inv = fit_gallery_open_set_calibrator(
                S_gg, yg, gal_bursts, EST_PI_NEW[sp], agg_kwargs
            )
            C_val, _ = aggregate_class_scores(S_qg, yg, u_cls, c_inv, **agg_kwargs)
            sp_preds = decode_predictions(C_val, u_cls, tau_opt, clf_new, feat_fn, decision_mode=decision_mode)

            for pos, p in zip(v_local_pos, sp_preds):
                preds_fold[pos] = p

        fold_preds[k] = preds_fold

    print(f"\n--- [{name}] ({time.time() - t0:.1f}s) ---")
    res = evaluate_cv_predictions(cv, train_df, fold_preds, verbose=True)
    res["name"] = name
    return res


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    X_all = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))[:len(train_df)]
    cv = build_or_load_cv(force_rebuild=True)

    results = []
    # 1. Decision Rule Ablation on Raw L2 Cosine
    results.append(run_experiment("01_RawL2_HardThresh", cv, train_df, X_all, transform_raw_l2, decision_mode="hard_thresh"))
    results.append(run_experiment("02_RawL2_ThreshElseRank2", cv, train_df, X_all, transform_raw_l2, decision_mode="thresh_else_rank2"))
    results.append(run_experiment("03_RawL2_BayesOptimalMAP5", cv, train_df, X_all, transform_raw_l2, decision_mode="bayes_opt"))

    # 2. Unsupervised Centering, Power-Law, & PCA-Whitening
    results.append(run_experiment("04_CenterL2_Inductive", cv, train_df, X_all, transform_center_l2_inductive))
    results.append(run_experiment("05_CenterL2_Transductive", cv, train_df, X_all, transform_center_l2_transductive))
    results.append(run_experiment("06_SignedPow0.75_CenterL2", cv, train_df, X_all, transform_power_center_l2, beta=0.75))
    results.append(run_experiment("07_PCA512_Whiten0.35", cv, train_df, X_all, transform_pca_whiten, n_comp=512, power=0.35))
    results.append(run_experiment("08_PCA768_Whiten0.50", cv, train_df, X_all, transform_pca_whiten, n_comp=768, power=0.50))

    # 3. Supervised Metric Learning (WCCN & Shrinkage LDA)
    results.append(run_experiment("09_WCCN_reg0.10_p0.50", cv, train_df, X_all, transform_wccn, reg=0.10, power=0.50))
    results.append(run_experiment("10_WCCN_reg0.05_p0.50", cv, train_df, X_all, transform_wccn, reg=0.05, power=0.50))
    results.append(run_experiment("11_WCCN_plus_ShrinkageLDA", cv, train_df, X_all, transform_lda_blend, shrinkage=0.25, w_lda=0.50))

    # 4. Class Aggregation, CSLS Hubness Reduction, & Query Expansion
    results.append(run_experiment("12_LDA_Blend_CSLS", cv, train_df, X_all, transform_lda_blend, shrinkage=0.25, w_lda=0.50, use_csls=True))
    results.append(run_experiment("13_LDA_Blend_LSE_sz0", cv, train_df, X_all, transform_lda_blend, shrinkage=0.25, w_lda=0.50, use_csls=True, agg_kwargs={"mode": "lse", "tau": 18.0, "size_pen": 0.0}))
    results.append(run_experiment("14_LDA_Blend_LSE_sz0.5", cv, train_df, X_all, transform_lda_blend, shrinkage=0.25, w_lda=0.50, use_csls=True, agg_kwargs={"mode": "lse", "tau": 18.0, "size_pen": 0.5}))
    results.append(run_experiment("15_LDA_Blend_CSLS_InductiveQE", cv, train_df, X_all, transform_lda_blend, shrinkage=0.25, w_lda=0.50, use_csls=True, qe_mode="inductive"))
    results.append(run_experiment("16_LDA_Blend_CSLS_TransductiveQE", cv, train_df, X_all, transform_lda_blend, shrinkage=0.25, w_lda=0.50, use_csls=True, qe_mode="transductive"))

    df_summary = pd.DataFrame(results)[["name", "lb_proxy_map5", "hyena_proxy_map5", "leopard_proxy_map5", "hyena_known_map5", "hyena_new_map5", "leopard_known_map5", "leopard_new_map5"]]
    print("\n================ EXP01 SUMMARY TABLE ================")
    print(df_summary.sort_values("lb_proxy_map5", ascending=False).to_string(index=False))
    df_summary.to_csv("cache/exp01_results.csv", index=False)
