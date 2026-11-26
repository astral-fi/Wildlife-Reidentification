"""
V3 Peak SIFT Engine (Anti-Static-Background-Trap + 8-DOF Homography + Epipolar + Foreground Area + Top-3/ID):
  - Sets BLAS/OpenMP/OpenCV to 1 thread per worker across 16 workers (eliminates thread thrashing -> ~3x faster!)
  - Uses AVX2-friendly fast top-2 argmax instead of scalar argpartition (~2x faster!)
  - Computes 12 geometric features per unique undirected pair:
      [0] n_r80            : symmetric mutual Lowe 0.82 count
      [1] n_r74            : avg directional Lowe 0.74 count
      [2] n_mut            : good mutual matches
      [3] margin_sum       : sum of s1 - s2 margins
      [4] n_sim_inl        : scale-bounded (0.30..3.30) 4-DOF Similarity RANSAC inliers (10px)
      [5] n_moving_sim_inl : Similarity inliers with pixel displacement >= 4.5px (kills static camera-trap bg!)
      [6] n_fg_sim_inl     : Similarity inliers inside [0.06..0.94]x[0.08..0.92] (kills corner timestamps/logos!)
      [7] inl_area_spread  : normalized spatial spread of Similarity inliers (kills tiny foliage patches!)
      [8] inl_mean_sim     : mean RootSIFT cosine of Similarity inliers (crisp spots vs mushy foliage!)
      [9] n_hom_inl        : non-degenerate 8-DOF Perspective Homography RANSAC inliers (12px, tilted flanks!)
      [10] n_fund_inl      : Epipolar Fundamental Matrix RANSAC inliers (6px, 3D out-of-plane rotation!)
      [11] sim_inlier_ratio: n_sim_inl / max(4, n_cand)
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import normalize

os.makedirs("cache_v3", exist_ok=True)

_OFFSETS = None
_ROOTSIFT = None
_XY = None
_BBOX = None  # (N, 4): [min_x, max_x, min_y, max_y]


def _init_worker():
    global _OFFSETS, _ROOTSIFT, _XY, _BBOX
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)
    _OFFSETS = np.load("cache/sift_offsets.npy", mmap_mode="r")
    _ROOTSIFT = np.load("cache/rootsift_desc_f16.npy", mmap_mode="r")
    _XY = np.load("cache/sift_xy.npy", mmap_mode="r")
    _BBOX = np.load("cache_v3/sift_img_bboxes.npy", mmap_mode="r")


def _fast_top2_rows(M):
    """SIMD-friendly exact top-2 per row of a C-contiguous float32 matrix (~2x faster than argpartition)."""
    n = M.shape[0]
    r = np.arange(n)
    i1 = np.argmax(M, axis=1)
    v1 = M[r, i1].copy()
    M[r, i1] = -2.0
    v2 = M.max(axis=1)
    M[r, i1] = v1
    return i1, v1, v2


def _compute_v3_pair_batch(pairs_arr):
    out = np.zeros((len(pairs_arr), 12), dtype=np.float32)
    for idx in range(len(pairs_arr)):
        i, j = int(pairs_arr[idx, 0]), int(pairs_arr[idx, 1])
        si, ei = int(_OFFSETS[i]), int(_OFFSETS[i + 1])
        sj, ej = int(_OFFSETS[j]), int(_OFFSETS[j + 1])
        ki, kj = ei - si, ej - sj
        if ki < 5 or kj < 5:
            continue

        di = np.asarray(_ROOTSIFT[si:ei], dtype=np.float32)
        dj = np.asarray(_ROOTSIFT[sj:ej], dtype=np.float32)
        S = di @ dj.T  # (ki, kj) C-contiguous

        nn_j, s1_i, s2_i = _fast_top2_rows(S)
        s1_i = np.clip(s1_i, -1.0, 1.0)
        s2_i = np.clip(s2_i, -1.0, 1.0)
        rat_i_sq = np.maximum(0.0, 2.0 - 2.0 * s1_i) / np.maximum(1e-7, 2.0 - 2.0 * s2_i)

        ST = np.ascontiguousarray(S.T)
        nn_i, s1_j, s2_j = _fast_top2_rows(ST)
        s1_j = np.clip(s1_j, -1.0, 1.0)
        s2_j = np.clip(s2_j, -1.0, 1.0)
        rat_j_sq = np.maximum(0.0, 2.0 - 2.0 * s1_j) / np.maximum(1e-7, 2.0 - 2.0 * s2_j)

        is_mut = (nn_i[nn_j] == np.arange(ki))
        sym_r80 = is_mut & (rat_i_sq < 0.82 ** 2) & (rat_j_sq[nn_j] < 0.82 ** 2)
        dir_r74 = 0.5 * (int((rat_i_sq < 0.74 ** 2).sum()) + int((rat_j_sq < 0.74 ** 2).sum()))
        good_mut = is_mut & ((rat_i_sq < 0.88 ** 2) | (rat_j_sq[nn_j] < 0.88 ** 2))

        n_r80 = int(sym_r80.sum())
        n_r74 = float(dir_r74)
        n_mut = int(good_mut.sum())
        margin_sum = float(np.sum(np.maximum(0.0, s1_i[good_mut] - s2_i[good_mut])))

        n_sim_inl = 0
        n_moving_inl = 0
        n_fg_inl = 0
        inl_area = 0.0
        inl_msim = 0.0
        n_hom_inl = 0
        n_fund_inl = 0

        n_cand = n_mut
        if n_cand >= 5 and (n_r80 >= 3 or n_r74 >= 3.0):
            pts_i = np.asarray(_XY[si:ei], dtype=np.float32)[good_mut]
            pts_j = np.asarray(_XY[sj:ej], dtype=np.float32)[nn_j[good_mut]]
            sims_cand = s1_i[good_mut]

            # 1) Scale-bounded 4-DOF Similarity RANSAC (10px)
            M, inl_s = cv2.estimateAffinePartial2D(
                pts_i, pts_j, method=cv2.RANSAC, ransacReprojThreshold=10.0, maxIters=180, confidence=0.985
            )
            if M is not None and inl_s is not None:
                sc = float(np.hypot(M[0, 0], M[1, 0]))
                if 0.30 <= sc <= 3.30:
                    im = (inl_s.ravel() > 0)
                    n_sim_inl = int(im.sum())
                    if n_sim_inl >= 3:
                        pi_in = pts_i[im]
                        pj_in = pts_j[im]
                        # Moving-animal displacement filter (kills static camera-trap background pixels!)
                        disp = np.hypot(pj_in[:, 0] - pi_in[:, 0], pj_in[:, 1] - pi_in[:, 1])
                        n_moving_inl = int((disp >= 4.5).sum())

                        # Normalized foreground interior filter (kills top/bottom timestamp banners!)
                        bxi = _BBOX[i]
                        bxj = _BBOX[j]
                        wi = max(10.0, float(bxi[1] - bxi[0]))
                        hi = max(10.0, float(bxi[3] - bxi[2]))
                        wj = max(10.0, float(bxj[1] - bxj[0]))
                        hj = max(10.0, float(bxj[3] - bxj[2]))
                        xi_n = (pi_in[:, 0] - bxi[0]) / wi
                        yi_n = (pi_in[:, 1] - bxi[2]) / hi
                        xj_n = (pj_in[:, 0] - bxj[0]) / wj
                        yj_n = (pj_in[:, 1] - bxj[2]) / hj
                        fg_m = (
                            (xi_n >= 0.06) & (xi_n <= 0.94) & (yi_n >= 0.08) & (yi_n <= 0.92)
                            & (xj_n >= 0.06) & (xj_n <= 0.94) & (yj_n >= 0.08) & (yj_n <= 0.92)
                        )
                        n_fg_inl = int(fg_m.sum())
                        inl_area = float(np.sqrt(max(1e-6, float(np.std(xi_n) * np.std(yi_n)))))
                        inl_msim = float(sims_cand[im].mean())

            # 2) 8-DOF Perspective Homography RANSAC (captures tilted planar flanks!)
            if n_cand >= 7 and (n_sim_inl >= 5 or n_r80 >= 6):
                H, inl_h = cv2.findHomography(pts_i, pts_j, method=cv2.RANSAC, ransacReprojThreshold=12.0, maxIters=180)
                if H is not None and inl_h is not None:
                    det = float(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0])
                    if 0.08 <= det <= 12.0:
                        n_hom_inl = int(inl_h.sum())

            # 3) Epipolar Fundamental Matrix RANSAC (captures 3D body rotation!)
            if n_cand >= 8 and (n_sim_inl >= 5 or n_hom_inl >= 6 or n_r80 >= 6):
                F, inl_f = cv2.findFundamentalMat(
                    pts_i, pts_j, method=cv2.FM_RANSAC, ransacReprojThreshold=6.0, confidence=0.985, maxIters=180
                )
                if F is not None and inl_f is not None:
                    n_fund_inl = int(inl_f.sum())

        sim_ratio = float(n_sim_inl) / max(4.0, float(n_cand))
        out[idx] = [
            n_r80, n_r74, n_mut, margin_sum,
            n_sim_inl, n_moving_inl, n_fg_inl, inl_area, inl_msim,
            n_hom_inl, n_fund_inl, sim_ratio
        ]
    return out


def build_image_bboxes():
    bb_path = "cache_v3/sift_img_bboxes.npy"
    if os.path.exists(bb_path):
        return np.load(bb_path)
    offsets = np.load("cache/sift_offsets.npy")
    xy = np.load("cache/sift_xy.npy", mmap_mode="r")
    N = len(offsets) - 1
    bb = np.zeros((N, 4), dtype=np.float32)
    for i in range(N):
        s, e = int(offsets[i]), int(offsets[i + 1])
        if e > s:
            pts = xy[s:e]
            bb[i] = [pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()]
    np.save(bb_path, bb)
    return bb


def build_sift_v3_csr(top_img_k=20, top_id_k=16, imgs_per_id=3):
    save_path = "cache_v3/sift_v3_peak_pairs_csr.npz"
    if os.path.exists(save_path):
        print("Loaded existing V3 Peak SIFT CSR cache:", save_path)
        return

    t0 = time.time()
    build_image_bboxes()

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
        # Blend both log1p spike-compressed & signed_pow MiewID for universal pair coverage!
        X_raw = miew_all[all_idx]
        X1 = normalize(np.sign(X_raw) * np.log1p(np.abs(X_raw) / 2.0), axis=1).astype(np.float64)
        X2 = normalize(np.sign(X_raw) * (np.abs(X_raw) ** 0.80), axis=1).astype(np.float64)
        X = normalize(0.5 * X1 + 0.5 * X2, axis=1)
        Xc = X - X.mean(0, keepdims=True)
        C = (Xc.T @ Xc) / len(Xc)
        ev, V = np.linalg.eigh(C)
        d = 768 if sp == "hyena" else 896
        Z = normalize((Xc @ V[:, -d:]) * ((ev[-d:] + 1.0e-3) ** (-0.31))[None, :], axis=1).astype(np.float32)

        S_all = Z @ Z.T
        np.fill_diagonal(S_all, -2.0)

        # 1) Top-20 overall neighbors (train + test -> captures all test-to-test bursts!)
        top_imgs = np.argpartition(-S_all, kth=top_img_k, axis=1)[:, :top_img_k]
        pair_a_list.append(np.repeat(all_idx, top_img_k))
        pair_b_list.append(all_idx[top_imgs.ravel()])

        # 2) Top-16 distinct training identities x best 3 images each (enables multi-image class SIFT voting!)
        n_tr_sp = len(tr_idx)
        S_to_tr = S_all[:, :n_tr_sp]
        yg = train_df["identity"].values[tr_idx]
        u_cls, c_inv = np.unique(yg, return_inverse=True)
        cols_c = [np.where(c_inv == c)[0] for c in range(len(u_cls))]

        C_max = np.empty((len(all_idx), len(u_cls)), dtype=np.float32)
        best3_loc = np.full((len(all_idx), len(u_cls), imgs_per_id), -1, dtype=np.int32)
        for c, cols in enumerate(cols_c):
            sub = S_to_tr[:, cols]
            if len(cols) == 1:
                C_max[:, c] = sub[:, 0]
                best3_loc[:, c, 0] = cols[0]
            else:
                k_take = min(imgs_per_id, len(cols))
                p = np.argpartition(-sub, kth=k_take - 1, axis=1)[:, :k_take]
                vals = np.take_along_axis(sub, p, axis=1)
                C_max[:, c] = vals.max(axis=1)
                best3_loc[:, c, :k_take] = cols[p]

        top_c = np.argpartition(-C_max, kth=top_id_k, axis=1)[:, :top_id_k]
        u2_l, v2_l = [], []
        for qi in range(len(all_idx)):
            g_locs = best3_loc[qi, top_c[qi]].ravel()
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
    print(f"Computing V3 Anti-BG + Homography + Epipolar SIFT for {len(uniq_pairs):,} unique pairs across 16 workers...")

    n_workers = 16
    n_chunks = 64
    chunks = np.array_split(uniq_pairs, n_chunks)
    res_ordered = [None] * n_chunks
    done_cnt = 0
    t_comp = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        fut_to_idx = {ex.submit(_compute_v3_pair_batch, ch): ci for ci, ch in enumerate(chunks)}
        for fut in as_completed(fut_to_idx):
            ci = fut_to_idx[fut]
            res_ordered[ci] = fut.result()
            done_cnt += 1
            if done_cnt % 16 == 0 or done_cnt == n_chunks:
                el = time.time() - t_comp
                print(f"  Progress: {done_cnt}/{n_chunks} chunks ({100*done_cnt/n_chunks:.0f}%) in {el:.1f}s")

    uniq_feats = np.vstack(res_ordered)  # (N_uniq, 12)

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
        f"(avg nbrs/img={counts.mean():.1f}, max_sim={uniq_feats[:,4].max():.0f}, "
        f"max_mov={uniq_feats[:,5].max():.0f}, max_hom={uniq_feats[:,9].max():.0f}, max_fund={uniq_feats[:,10].max():.0f})"
    )


if __name__ == "__main__":
    build_sift_v3_csr(top_img_k=20, top_id_k=16, imgs_per_id=3)
