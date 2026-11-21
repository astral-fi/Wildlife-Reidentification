import os
import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import normalize

_OFFSETS = None
_ROOTSIFT = None
_XY = None


def _init_worker():
    global _OFFSETS, _ROOTSIFT, _XY
    _OFFSETS = np.load("cache/sift_offsets.npy", mmap_mode="r")
    _ROOTSIFT = np.load("cache/rootsift_desc_f16.npy", mmap_mode="r")
    _XY = np.load("cache/sift_xy.npy", mmap_mode="r")


def _compute_pair_batch(pairs_arr):
    out = np.zeros((len(pairs_arr), 6), dtype=np.float32)
    for idx in range(len(pairs_arr)):
        i, j = int(pairs_arr[idx, 0]), int(pairs_arr[idx, 1])
        si, ei = _OFFSETS[i], _OFFSETS[i + 1]
        sj, ej = _OFFSETS[j], _OFFSETS[j + 1]
        ki, kj = ei - si, ej - sj
        if ki < 4 or kj < 4:
            continue
        di = np.asarray(_ROOTSIFT[si:ei], dtype=np.float32)
        dj = np.asarray(_ROOTSIFT[sj:ej], dtype=np.float32)
        S = di @ dj.T  # (ki, kj)

        # Top-2 matches for i in j
        p_idx = np.argpartition(-S, kth=1, axis=1)[:, :2]
        s_pair = np.take_along_axis(S, p_idx, axis=1)
        swap = s_pair[:, 1] > s_pair[:, 0]
        if np.any(swap):
            s_pair[swap] = s_pair[swap, ::-1]
            p_idx[swap] = p_idx[swap, ::-1]

        s1 = np.clip(s_pair[:, 0], -1.0, 1.0)
        s2 = np.clip(s_pair[:, 1], -1.0, 1.0)
        nn_j = p_idx[:, 0]

        d1_sq = np.maximum(0.0, 2.0 - 2.0 * s1)
        d2_sq = np.maximum(1e-7, 2.0 - 2.0 * s2)
        ratio_sq = d1_sq / d2_sq

        m80 = ratio_sq < (0.82 ** 2)
        m72 = ratio_sq < (0.74 ** 2)
        best_i_for_j = np.argmax(S, axis=0)
        mutual = (best_i_for_j[nn_j] == np.arange(ki)) & (ratio_sq < (0.88 ** 2))

        n_r80 = int(m80.sum())
        n_r72 = int(m72.sum())
        n_mut = int(mutual.sum())
        margin_sum = float(np.sum(np.maximum(0.0, s1[mutual] - s2[mutual])))

        # Only run OpenCV RANSAC when >= 5 strong correspondences exist!
        cand_mask = mutual | m80
        n_cand = int(cand_mask.sum())
        n_inliers = 0
        if n_cand >= 5:
            pts_i = np.asarray(_XY[si:ei])[cand_mask]
            pts_j = np.asarray(_XY[sj:ej])[nn_j[cand_mask]]
            _, inl = cv2.estimateAffinePartial2D(
                pts_i, pts_j, method=cv2.RANSAC, ransacReprojThreshold=15.0, maxIters=200, confidence=0.985
            )
            if inl is not None:
                n_inliers = int(inl.sum())

        inlier_ratio = float(n_inliers) / max(4.0, float(n_cand))
        out[idx] = [n_r80, n_r72, n_mut, margin_sum, n_inliers, inlier_ratio]
    return out


def build_global_sift_pair_cache(top_k=36):
    save_path = f"cache/sift_top{top_k}_global.npz"
    if os.path.exists(save_path):
        print("Loaded existing SIFT global pair cache:", save_path)
        return
    t0 = time.time()
    train_df = pd.read_csv("reid_student_data_main/train.csv")
    test_df = pd.read_csv("reid_student_data_main/test.csv")
    species_all = np.concatenate([train_df["species"].values, test_df["species"].values])
    miew_all = np.load("reid_student_data_main/miewid_features.npy")
    N = len(species_all)

    # Compute species-whitened cosine across all 9,954 images to select top_k candidate neighbors per image!
    nbr_idx = np.full((N, top_k), -1, dtype=np.int32)
    for sp in ["hyena", "leopard"]:
        idx = np.where(species_all == sp)[0]
        X = normalize(np.sign(miew_all[idx]) * (np.abs(miew_all[idx]) ** 0.85), axis=1).astype(np.float64)
        Xc = X - X.mean(axis=0, keepdims=True)
        C = (Xc.T @ Xc) / len(Xc)
        eigvals, V = np.linalg.eigh(C)
        m = eigvals > 1e-8
        Z = normalize((Xc @ V[:, m]) * ((eigvals[m] + 2e-3) ** (-0.325))[None, :], axis=1).astype(np.float32)
        S = Z @ Z.T
        np.fill_diagonal(S, -2.0)
        top_local = np.argsort(-S, axis=1)[:, :top_k]
        nbr_idx[idx] = idx[top_local]

    # Deduplicate unordered pairs (min(i,j), max(i,j))!
    u_rows = np.repeat(np.arange(N, dtype=np.int32), top_k)
    v_rows = nbr_idx.ravel()
    a = np.minimum(u_rows, v_rows)
    b = np.maximum(u_rows, v_rows)
    pair_keys = a.astype(np.int64) * N + b.astype(np.int64)
    uniq_keys, inv_map = np.unique(pair_keys, return_inverse=True)
    uniq_pairs = np.column_stack([uniq_keys // N, uniq_keys % N]).astype(np.int32)
    print(f"Computing SIFT+RANSAC once for {len(uniq_pairs):,} unique candidate pairs across 16 processes...")

    n_workers = 16
    chunks = np.array_split(uniq_pairs, n_workers * 4)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        res_list = list(ex.map(_compute_pair_batch, chunks))

    uniq_feats = np.vstack(res_list)  # (N_uniq, 6)
    nbr_feats = uniq_feats[inv_map].reshape(N, top_k, 6)
    np.savez_compressed(save_path, nbr_idx=nbr_idx, nbr_feats=nbr_feats)
    print(f"Saved {save_path} in {time.time() - t0:.1f}s! (Max RANSAC inliers observed: {uniq_feats[:, 4].max():.0f})")


if __name__ == "__main__":
    build_global_sift_pair_cache(top_k=36)
