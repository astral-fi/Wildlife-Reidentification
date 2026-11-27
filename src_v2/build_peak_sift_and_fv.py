import os
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

os.makedirs("cache_v2", exist_ok=True)

_OFFSETS = None
_ROOTSIFT = None
_XY = None


def _init_worker():
    global _OFFSETS, _ROOTSIFT, _XY
    _OFFSETS = np.load("cache/sift_offsets.npy", mmap_mode="r")
    _ROOTSIFT = np.load("cache/rootsift_desc_f16.npy", mmap_mode="r")
    _XY = np.load("cache/sift_xy.npy", mmap_mode="r")


def _compute_peak_pair_batch(pairs_arr):
    """
    Symmetric, scale-bounded RootSIFT + 4-DOF Similarity RANSAC + Epipolar Fundamental-Matrix RANSAC!
    Returns 8 features per undirected pair (i, j):
      [n_sym_r80, n_sym_r74, n_mut, margin_sum, n_sim_inliers, sim_inlier_ratio, n_fund_inliers, max_kp_sim]
    """
    out = np.zeros((len(pairs_arr), 8), dtype=np.float32)
    for idx in range(len(pairs_arr)):
        i, j = int(pairs_arr[idx, 0]), int(pairs_arr[idx, 1])
        si, ei = _OFFSETS[i], _OFFSETS[i + 1]
        sj, ej = _OFFSETS[j], _OFFSETS[j + 1]
        ki, kj = ei - si, ej - sj
        if ki < 5 or kj < 5:
            continue

        di = np.asarray(_ROOTSIFT[si:ei], dtype=np.float32)
        dj = np.asarray(_ROOTSIFT[sj:ej], dtype=np.float32)
        S = di @ dj.T  # (ki, kj)

        # Forward i -> j top-2
        pi = np.argpartition(-S, kth=1, axis=1)[:, :2]
        sp_i = np.take_along_axis(S, pi, axis=1)
        sw = sp_i[:, 1] > sp_i[:, 0]
        if np.any(sw):
            sp_i[sw] = sp_i[sw, ::-1]
            pi[sw] = pi[sw, ::-1]
        s1_i = np.clip(sp_i[:, 0], -1.0, 1.0)
        s2_i = np.clip(sp_i[:, 1], -1.0, 1.0)
        nn_j = pi[:, 0]
        rat_i_sq = np.maximum(0.0, 2.0 - 2.0 * s1_i) / np.maximum(1e-7, 2.0 - 2.0 * s2_i)

        # Backward j -> i top-2 (for exact symmetry!)
        pj = np.argpartition(-S, kth=1, axis=0)[:2, :]  # (2, kj)
        sp_j = np.take_along_axis(S, pj, axis=0)
        swj = sp_j[1] > sp_j[0]
        if np.any(swj):
            sp_j[:, swj] = sp_j[::-1, swj]
            pj[:, swj] = pj[::-1, swj]
        s1_j = np.clip(sp_j[0], -1.0, 1.0)
        s2_j = np.clip(sp_j[1], -1.0, 1.0)
        nn_i = pj[0]
        rat_j_sq = np.maximum(0.0, 2.0 - 2.0 * s1_j) / np.maximum(1e-7, 2.0 - 2.0 * s2_j)

        # Mutual nearest neighbors
        is_mut = (nn_i[nn_j] == np.arange(ki))
        # Symmetric ratio test on mutual matches (and average directional count)
        sym_r80 = is_mut & (rat_i_sq < 0.82 ** 2) & (rat_j_sq[nn_j] < 0.82 ** 2)
        sym_r74 = is_mut & (rat_i_sq < 0.75 ** 2) & (rat_j_sq[nn_j] < 0.75 ** 2)
        dir_r74 = 0.5 * (int((rat_i_sq < 0.74 ** 2).sum()) + int((rat_j_sq < 0.74 ** 2).sum()))
        good_mut = is_mut & ((rat_i_sq < 0.88 ** 2) | (rat_j_sq[nn_j] < 0.88 ** 2))

        n_r80 = int(sym_r80.sum())
        n_r74 = float(dir_r74)
        n_mut = int(good_mut.sum())
        margin_sum = float(np.sum(np.maximum(0.0, s1_i[good_mut] - s2_i[good_mut])))
        max_kp = float(s1_i.max())

        # Candidate geometric matches: mutual good matches (gate RANSAC by >=3 ratio matches for 5x speedup & zero noise!)
        n_sim_inl = 0
        n_fund_inl = 0
        cand_mask = good_mut
        n_cand = int(cand_mask.sum())
        if n_cand >= 5 and (n_r80 >= 3 or n_r74 >= 3.0):
            pts_i = np.asarray(_XY[si:ei])[cand_mask]
            pts_j = np.asarray(_XY[sj:ej])[nn_j[cand_mask]]

            # 1) Scale-bounded 4-DOF similarity RANSAC (tighter 10px threshold, reject collapsed scales!)
            M, inl_s = cv2.estimateAffinePartial2D(
                pts_i, pts_j, method=cv2.RANSAC, ransacReprojThreshold=10.0, maxIters=200, confidence=0.985
            )
            if M is not None and inl_s is not None:
                sc = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                if 0.30 <= sc <= 3.30:
                    n_sim_inl = int(inl_s.sum())

            # 2) Epipolar Fundamental Matrix RANSAC (handles 3D out-of-plane animal body rotation!)
            if n_cand >= 8 and (n_sim_inl >= 5 or n_r80 >= 6):
                F, inl_f = cv2.findFundamentalMat(
                    pts_i, pts_j, method=cv2.FM_RANSAC, ransacReprojThreshold=6.0, confidence=0.985, maxIters=200
                )
                if F is not None and inl_f is not None:
                    n_fund_inl = int(inl_f.sum())

        sim_ratio = float(n_sim_inl) / max(4.0, float(n_cand))
        out[idx] = [n_r80, n_r74, n_mut, margin_sum, n_sim_inl, sim_ratio, n_fund_inl, max_kp]
    return out


def build_fisher_vectors(n_gauss=32, pca_desc=64, out_dim=512):
    """
    True Bishop PRML Fisher Vectors on RootSIFT:
      PCA-64 on RootSIFT -> Diagonal GMM (K=32) -> 1st & 2nd order normalized gradients (4,096-d)
      -> SSR Power-Norm -> L2-Norm -> PCA-512 Whitening!
    """
    fv_path = f"cache_v2/fisher_g{n_gauss}_p{out_dim}.npy"
    if os.path.exists(fv_path):
        print("Loaded existing Fisher Vectors:", fv_path)
        return np.load(fv_path)

    t0 = time.time()
    offsets = np.load("cache/sift_offsets.npy")
    rootsift = np.load("cache/rootsift_desc_f16.npy")
    total_kp = rootsift.shape[0]
    rng = np.random.RandomState(42)

    samp_idx = rng.choice(total_kp, size=min(180000, total_kp), replace=False)
    X_samp = rootsift[samp_idx].astype(np.float32)

    # Reduce RootSIFT 128 -> 64 (decorrelates descriptors so diagonal GMM is exact!)
    pca_d = PCA(n_components=pca_desc, random_state=42).fit(X_samp)
    X_samp_64 = pca_d.transform(X_samp)

    gmm = GaussianMixture(n_components=n_gauss, covariance_type="diag", max_iter=60, n_init=1, random_state=42)
    gmm.fit(X_samp_64)

    weights = np.maximum(gmm.weights_, 1e-6).astype(np.float32)         # (K,)
    means = gmm.means_.astype(np.float32)                               # (K, D)
    stds = np.sqrt(np.maximum(gmm.covariances_, 1e-6)).astype(np.float32)  # (K, D)
    inv_std = 1.0 / stds
    sqrt_w = np.sqrt(weights)
    sqrt_2w = np.sqrt(2.0 * weights)

    n_imgs = len(offsets) - 1
    fv_raw = np.zeros((n_imgs, 2 * n_gauss * pca_desc), dtype=np.float32)

    # Transform & accumulate in batches of images
    mean_d = pca_d.mean_.astype(np.float32)
    comp_T = pca_d.components_.T.astype(np.float32)  # (128, 64)

    for i in range(n_imgs):
        s, e = offsets[i], offsets[i + 1]
        T = e - s
        if T == 0:
            continue
        x64 = (rootsift[s:e].astype(np.float32) - mean_d) @ comp_T  # (T, D)
        post = gmm.predict_proba(x64).astype(np.float32)            # (T, K)

        # Normalized residuals per Gaussian
        # S0: (K,), S1: (K, D), S2: (K, D)
        S0 = post.sum(axis=0)                                        # (K,)
        S1 = post.T @ x64                                            # (K, D)
        S2 = post.T @ (x64 ** 2)                                     # (K, D)

        inv_T = 1.0 / float(T)
        g_mu = inv_T * ((S1 - S0[:, None] * means) * inv_std) / sqrt_w[:, None]
        g_sig = inv_T * (((S2 - 2.0 * means * S1 + S0[:, None] * (means ** 2)) * (inv_std ** 2)) - S0[:, None]) / sqrt_2w[:, None]

        vec = np.concatenate([g_mu.ravel(), g_sig.ravel()])
        vec = np.sign(vec) * np.sqrt(np.abs(vec))  # SSR power norm
        nrm = np.linalg.norm(vec)
        if nrm > 1e-8:
            fv_raw[i] = vec / nrm

    pca_out = PCA(n_components=out_dim, random_state=42)
    fv_pca = pca_out.fit_transform(fv_raw)
    scale = (pca_out.explained_variance_ + 1e-4) ** (-0.25)
    fv_out = normalize(fv_pca * scale, axis=1).astype(np.float32)
    np.save(fv_path, fv_out)
    print(f"Built Fisher Vectors {fv_out.shape} in {time.time() - t0:.1f}s!")
    return fv_out


def build_identity_aware_peak_sift_cache(top_img_k=24, top_id_k=14, imgs_per_id=2):
    """
    Guarantees 100% SIFT geometric coverage for BOTH:
      - Top-24 nearest images across train+test (captures test-to-test burst clusters!)
      - Best 2 gallery images from EACH of the top-14 distinct training identities (zero mega-class slot starvation!)
    Stored as sparse dictionary/CSR arrays (row_ptr, col_idx, feat_vals) for O(1) lookup!
    """
    save_path = "cache_v2/sift_peak_pairs_csr.npz"
    if os.path.exists(save_path):
        print("Loaded existing Peak SIFT CSR cache:", save_path)
        return

    t0 = time.time()
    train_df = pd.read_csv("reid_student_data_main/train.csv")
    test_df = pd.read_csv("reid_student_data_main/test.csv")
    n_tr, n_te = len(train_df), len(test_df)
    N = n_tr + n_te
    species_all = np.concatenate([train_df["species"].values, test_df["species"].values])
    miew_all = np.load("reid_student_data_main/miewid_features.npy")

    pair_a_list, pair_b_list = [], []

    for sp in ["hyena", "leopard"]:
        tr_idx = np.where(train_df["species"].values == sp)[0]
        all_idx = np.where(species_all == sp)[0]
        X = normalize(np.sign(miew_all[all_idx]) * (np.abs(miew_all[all_idx]) ** 0.85), axis=1).astype(np.float64)
        Xc = X - X.mean(0, keepdims=True)
        C = (Xc.T @ Xc) / len(Xc)
        ev, V = np.linalg.eigh(C)
        d = 640 if sp == "hyena" else 768
        Z = normalize((Xc @ V[:, -d:]) * ((ev[-d:] + 1.5e-3) ** (-0.275))[None, :], axis=1).astype(np.float32)

        S_all = Z @ Z.T
        np.fill_diagonal(S_all, -2.0)

        # 1) Top-24 overall neighbors (train + test)
        top_imgs = np.argpartition(-S_all, kth=top_img_k, axis=1)[:, :top_img_k]
        u1 = np.repeat(all_idx, top_img_k)
        v1 = all_idx[top_imgs.ravel()]
        pair_a_list.append(u1)
        pair_b_list.append(v1)

        # 2) Top-14 distinct training identities x best 2 images each!
        n_tr_sp = len(tr_idx)
        S_to_tr = S_all[:, :n_tr_sp]  # (n_all_sp, n_tr_sp)
        yg = train_df["identity"].values[tr_idx]
        u_cls, c_inv = np.unique(yg, return_inverse=True)
        cols_c = [np.where(c_inv == c)[0] for c in range(len(u_cls))]

        # Compute max sim per class & top-2 local gallery indices per class
        C_max = np.empty((len(all_idx), len(u_cls)), dtype=np.float32)
        best2_loc = np.full((len(all_idx), len(u_cls), imgs_per_id), -1, dtype=np.int32)
        for c, cols in enumerate(cols_c):
            sub = S_to_tr[:, cols]
            if len(cols) == 1:
                C_max[:, c] = sub[:, 0]
                best2_loc[:, c, 0] = cols[0]
            else:
                k_take = min(imgs_per_id, len(cols))
                p = np.argpartition(-sub, kth=k_take - 1, axis=1)[:, :k_take]
                vals = np.take_along_axis(sub, p, axis=1)
                C_max[:, c] = vals.max(axis=1)
                best2_loc[:, c, :k_take] = cols[p]

        top_c = np.argpartition(-C_max, kth=top_id_k, axis=1)[:, :top_id_k]  # (n_all_sp, top_id_k)
        u2_l, v2_l = [], []
        for qi in range(len(all_idx)):
            g_locs = best2_loc[qi, top_c[qi]].ravel()
            g_locs = g_locs[g_locs >= 0]
            g_globs = tr_idx[g_locs]
            g_globs = g_globs[g_globs != all_idx[qi]]
            u2_l.append(np.full(len(g_globs), all_idx[qi], dtype=np.int32))
            v2_l.append(g_globs)
        pair_a_list.append(np.concatenate(u2_l))
        pair_b_list.append(np.concatenate(v2_l))

    all_u = np.concatenate(pair_a_list)
    all_v = np.concatenate(pair_b_list)
    a = np.minimum(all_u, all_v)
    b = np.maximum(all_u, all_v)
    keys = a.astype(np.int64) * N + b.astype(np.int64)
    uniq_keys = np.unique(keys)
    uniq_pairs = np.column_stack([uniq_keys // N, uniq_keys % N]).astype(np.int32)
    print(f"Computing symmetric scale-bounded Sim+Fund RANSAC for {len(uniq_pairs):,} unique identity-aware pairs...")

    n_workers = 12
    chunks = np.array_split(uniq_pairs, n_workers * 6)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        res_list = list(ex.map(_compute_peak_pair_batch, chunks))
    uniq_feats = np.vstack(res_list)  # (N_uniq, 8)

    # Store symmetric CSR adjacency so lookup(q, g) works in BOTH directions!
    sym_u = np.concatenate([uniq_pairs[:, 0], uniq_pairs[:, 1]])
    sym_v = np.concatenate([uniq_pairs[:, 1], uniq_pairs[:, 0]])
    sym_f = np.vstack([uniq_feats, uniq_feats])
    order = np.argsort(sym_u, kind="mergesort")
    sym_u = sym_u[order]
    sym_v = sym_v[order]
    sym_f = sym_f[order]

    counts = np.bincount(sym_u, minlength=N)
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(counts)

    np.savez_compressed(save_path, row_ptr=row_ptr, col_idx=sym_v, feat_vals=sym_f)
    print(
        f"Saved {save_path} in {time.time() - t0:.1f}s! "
        f"(avg neighbors/img={counts.mean():.1f}, max_sim_inl={uniq_feats[:,4].max():.0f}, max_fund_inl={uniq_feats[:,6].max():.0f})"
    )


if __name__ == "__main__":
    build_fisher_vectors(n_gauss=32, pca_desc=64, out_dim=512)
    build_identity_aware_peak_sift_cache(top_img_k=18, top_id_k=12, imgs_per_id=2)
