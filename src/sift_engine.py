import os
import time
import cv2
import faiss
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def load_rootsift_cache(cache_dir="cache"):
    offsets = np.load(os.path.join(cache_dir, "sift_offsets.npy"))
    desc_u8 = np.load(os.path.join(cache_dir, "sift_desc.npy"))
    xy = np.load(os.path.join(cache_dir, "sift_xy.npy"))

    # RootSIFT: L1 normalize -> sqrt -> L2 normalize
    rs_path = os.path.join(cache_dir, "rootsift_desc_f16.npy")
    if os.path.exists(rs_path):
        rootsift = np.load(rs_path)
    else:
        f32 = desc_u8.astype(np.float32)
        l1 = np.sum(f32, axis=1, keepdims=True) + 1e-7
        f32 = np.sqrt(f32 / l1)
        rootsift = normalize(f32, axis=1).astype(np.float16)
        np.save(rs_path, rootsift)
    return offsets, rootsift, xy


def pair_sift_geom_stats(q_desc, q_xy, g_desc, g_xy):
    """
    Fast pairwise RootSIFT matching + Lowe's ratio test + Mutual NN + 4-DOF RANSAC geometric verification.
    Returns 7 scalar features:
      [n_ratio_80, n_ratio_72, n_mutual, sum_margin, n_inliers_sim, inlier_ratio, max_sim_kp]
    """
    kq, kg = q_desc.shape[0], g_desc.shape[0]
    if kq < 4 or kg < 4:
        return np.zeros(7, dtype=np.float32)

    # Cosine similarity matrix (kq, kg) in float32
    S = q_desc.astype(np.float32) @ g_desc.astype(np.float32).T
    # Top-2 gallery matches for each query kp
    part_idx = np.argpartition(-S, kth=1, axis=1)[:, :2]
    s_pair = np.take_along_axis(S, part_idx, axis=1)
    swap = s_pair[:, 1] > s_pair[:, 0]
    if np.any(swap):
        s_pair[swap] = s_pair[swap, ::-1]
        part_idx[swap] = part_idx[swap, ::-1]

    s1 = np.clip(s_pair[:, 0], -1.0, 1.0)
    s2 = np.clip(s_pair[:, 1], -1.0, 1.0)
    nn_g = part_idx[:, 0]

    # Euclidean distance squared = 2 - 2*s
    d1_sq = np.maximum(0.0, 2.0 - 2.0 * s1)
    d2_sq = np.maximum(1e-7, 2.0 - 2.0 * s2)
    ratio_sq = d1_sq / d2_sq

    mask_80 = ratio_sq < (0.82 ** 2)
    mask_72 = ratio_sq < (0.74 ** 2)

    # Mutual NN check
    best_q_for_g = np.argmax(S, axis=0)  # (kg,)
    mutual = (best_q_for_g[nn_g] == np.arange(kq))
    good_mut = mutual & (ratio_sq < (0.88 ** 2))

    n_r80 = int(mask_80.sum())
    n_r72 = int(mask_72.sum())
    n_mut = int(good_mut.sum())
    margin_sum = float(np.sum(np.maximum(0.0, s1[good_mut] - s2[good_mut])))
    max_kp = float(s1.max())

    # Geometric verification via 4-DOF similarity RANSAC (scale + rotation + translation)
    # Use union of good_mut and mask_80 for candidate correspondences
    cand_mask = good_mut | mask_80
    n_cand = int(cand_mask.sum())
    n_inliers = 0
    if n_cand >= 4:
        pts_q = q_xy[cand_mask]
        pts_g = g_xy[nn_g[cand_mask]]
        _, inliers = cv2.estimateAffinePartial2D(
            pts_q, pts_g,
            method=cv2.RANSAC,
            ransacReprojThreshold=15.0,
            maxIters=300,
            confidence=0.99
        )
        if inliers is not None:
            n_inliers = int(inliers.sum())

    inlier_ratio = float(n_inliers) / max(4.0, float(n_cand))
    return np.array([n_r80, n_r72, n_mut, margin_sum, n_inliers, inlier_ratio, max_kp], dtype=np.float32)


def batch_compute_sift_pair_features(query_global_rows, cand_global_rows_2d, offsets, rootsift, xy, n_workers=16):
    """
    query_global_rows: (N_q,) int indices in [0..9953]
    cand_global_rows_2d: (N_q, M) int indices in [0..9953]
    Returns: (N_q, M, 7) float32 array of pairwise SIFT + RANSAC features.
    """
    N_q, M = cand_global_rows_2d.shape
    out = np.zeros((N_q, M, 7), dtype=np.float32)

    def worker(q_range):
        for qi in q_range:
            q_row = query_global_rows[qi]
            qs, qe = offsets[q_row], offsets[q_row + 1]
            if qe - qs < 4:
                continue
            qd = rootsift[qs:qe]
            qx = xy[qs:qe]
            for m in range(M):
                g_row = cand_global_rows_2d[qi, m]
                if g_row < 0:
                    continue
                gs, ge = offsets[g_row], offsets[g_row + 1]
                if ge - gs < 4:
                    continue
                out[qi, m] = pair_sift_geom_stats(qd, qx, rootsift[gs:ge], xy[gs:ge])

    step = max(1, (N_q + n_workers * 4 - 1) // (n_workers * 4))
    ranges = [range(i, min(N_q, i + step)) for i in range(0, N_q, step)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(worker, ranges))
    return out


def build_or_load_vlad_embeddings(offsets, rootsift, n_clusters=64, pca_dim=512, cache_dir="cache"):
    """
    Unsupervised VLAD (Vector of Locally Aggregated Descriptors, Jegou et al. CVPR 2010)
    built with k-means + intra-normalization + SSR power-law + PCA whitening on RootSIFT.
    """
    vlad_path = os.path.join(cache_dir, f"vlad_c{n_clusters}_p{pca_dim}.npy")
    if os.path.exists(vlad_path):
        return np.load(vlad_path)

    t0 = time.time()
    rng = np.random.RandomState(42)
    # Sample 250,000 descriptors to fit k-means codebook via FAISS / MiniBatchKMeans
    total_kp = rootsift.shape[0]
    samp_idx = rng.choice(total_kp, size=min(250000, total_kp), replace=False)
    X_samp = rootsift[samp_idx].astype(np.float32)

    kmeans = faiss.Kmeans(d=128, k=n_clusters, niter=25, verbose=False, seed=42)
    kmeans.train(X_samp)
    centroids = kmeans.centroids  # (K, 128)
    index = faiss.IndexFlatL2(128)
    index.add(centroids)

    n_imgs = len(offsets) - 1
    vlad_raw = np.zeros((n_imgs, n_clusters * 128), dtype=np.float32)

    # Assign in chunks of 500k keypoints for speed
    all_assign = np.empty(total_kp, dtype=np.int32)
    chunk = 500000
    for s in range(0, total_kp, chunk):
        e = min(total_kp, s + chunk)
        _, I = index.search(rootsift[s:e].astype(np.float32), 1)
        all_assign[s:e] = I[:, 0]

    for i in range(n_imgs):
        s, e = offsets[i], offsets[i + 1]
        if e == s:
            continue
        desc_i = rootsift[s:e].astype(np.float32)
        c_i = all_assign[s:e]
        v = np.zeros((n_clusters, 128), dtype=np.float32)
        # Sum residuals (x - mu_k) per cluster
        np.add.at(v, c_i, desc_i - centroids[c_i])
        # Intra-normalization + SSR power-law (alpha=0.5) + global L2 norm
        v = normalize(v, axis=1)
        v_flat = np.sign(v.ravel()) * np.sqrt(np.abs(v.ravel()))
        vlad_raw[i] = normalize(v_flat.reshape(1, -1), axis=1)[0]

    pca = PCA(n_components=pca_dim, random_state=42)
    vlad_pca = pca.fit_transform(vlad_raw)
    scale = (pca.explained_variance_ + 1e-4) ** (-0.25)
    vlad_out = normalize(vlad_pca * scale, axis=1).astype(np.float32)
    np.save(vlad_path, vlad_out)
    print(f"Built VLAD ({vlad_out.shape}) in {time.time() - t0:.1f}s")
    return vlad_out
