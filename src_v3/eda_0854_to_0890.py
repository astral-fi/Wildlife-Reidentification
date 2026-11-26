import os
import shutil
import pickle
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize

os.makedirs("src_v3", exist_ok=True)
os.makedirs("cache_v3", exist_ok=True)

# 1. Preserve verified 0.854 submission backup immediately!
if not os.path.exists("submissions_v2/submission_0854_verified_backup.csv"):
    shutil.copy2("submission.csv", "submissions_v2/submission_0854_verified_backup.csv")
    print("Saved verified 0.854 backup -> submissions_v2/submission_0854_verified_backup.csv")

train_df = pd.read_csv("reid_student_data_main/train.csv")
test_df = pd.read_csv("reid_student_data_main/test.csv")
miew_ids = pd.read_csv("reid_student_data_main/miewid_ids.csv")
sub_854 = pd.read_csv("submissions_v2/submission_0854_verified_backup.csv")
sub_829 = pd.read_csv("submissions/submission_best.csv")

print("\n=== 1. METADATA & FILE STRUCTURE DEEP AUDIT ===")
print("train_df cols:", list(train_df.columns), "shape:", train_df.shape)
print("test_df  cols:", list(test_df.columns), "shape:", test_df.shape)
print("miewid_ids cols:", list(miew_ids.columns), "shape:", miew_ids.shape)
print("miewid_ids head:\n", miew_ids.head(3))
print("train_df head:\n", train_df.head(3))
print("test_df  head:\n", test_df.head(3))

# Check if test_df row ordering has temporal/adjacent correlation!
miew_all = np.load("reid_student_data_main/miewid_features.npy")
X_norm = normalize(miew_all, axis=1)
n_tr = len(train_df)
X_tr = X_norm[:n_tr]
X_te = X_norm[n_tr:]

adj_tr = np.sum(X_tr[:-1] * X_tr[1:], axis=1)
adj_te = np.sum(X_te[:-1] * X_te[1:], axis=1)
same_id_adj = (train_df["identity"].values[:-1] == train_df["identity"].values[1:]).mean()
same_sp_adj_tr = (train_df["species"].values[:-1] == train_df["species"].values[1:]).mean()
same_sp_adj_te = (test_df["species"].values[:-1] == test_df["species"].values[1:]).mean()
print(f"\nAdjacent-row cosine check: train mean={adj_tr.mean():.4f} (same_id={same_id_adj:.4f}, same_sp={same_sp_adj_tr:.4f}) | test mean={adj_te.mean():.4f} (same_sp={same_sp_adj_te:.4f})")

# Check MiewID feature distribution (skew, heavy tails, zero/low-var dims, norm correlation with quality!)
print("\n=== 2. MIEWID EMBEDDING SPECTRUM & QUALITY AUDIT ===")
print("miew_all dtype:", miew_all.dtype, "min/max/mean/std:", miew_all.min(), miew_all.max(), miew_all.mean(), miew_all.std())
dim_std = miew_all.std(axis=0)
print("miew_all dim std quantiles [0, 1%, 5%, 50%, 95%, 99%, 100%]:", np.round(np.quantile(dim_std, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]), 3))
norms = np.linalg.norm(miew_all, axis=1)
for sp in ["hyena", "leopard"]:
    mtr = train_df["species"].values == sp
    mte = test_df["species"].values == sp
    print(f"  [{sp:7s}] L2 norm train={norms[:n_tr][mtr].mean():.2f}+/-{norms[:n_tr][mtr].std():.2f} | test={norms[n_tr:][mte].mean():.2f}+/-{norms[n_tr:][mte].std():.2f}")

# Check SIFT XY spatial bounding-box & keypoint density stats!
print("\n=== 3. SIFT SPATIAL GEOMETRY (XY) & KEYPOINT DISTRIBUTION AUDIT ===")
offsets = np.load("cache/sift_offsets.npy")
xy = np.load("cache/sift_xy.npy", mmap_mode="r")
kps = np.diff(offsets)
max_x = np.zeros(len(kps), dtype=np.float32)
max_y = np.zeros(len(kps), dtype=np.float32)
min_x = np.zeros(len(kps), dtype=np.float32)
min_y = np.zeros(len(kps), dtype=np.float32)
for i in range(len(kps)):
    s, e = offsets[i], offsets[i + 1]
    if e > s:
        pts = xy[s:e]
        min_x[i], min_y[i] = pts[:, 0].min(), pts[:, 1].min()
        max_x[i], max_y[i] = pts[:, 0].max(), pts[:, 1].max()
aspect = (max_x - min_x + 1.0) / (max_y - min_y + 1.0)
for sp in ["hyena", "leopard"]:
    mtr = train_df["species"].values == sp
    mte = test_df["species"].values == sp
    print(
        f"  [{sp:7s}] kp_cnt tr={np.median(kps[:n_tr][mtr]):.0f}/te={np.median(kps[n_tr:][mte]):.0f} | "
        f"max_x tr={np.median(max_x[:n_tr][mtr]):.0f}/te={np.median(max_x[n_tr:][mte]):.0f} | "
        f"max_y tr={np.median(max_y[:n_tr][mtr]):.0f}/te={np.median(max_y[n_tr:][mte]):.0f} | "
        f"aspect tr={np.median(aspect[:n_tr][mtr]):.2f}/te={np.median(aspect[n_tr:][mte]):.2f}"
    )

# Check how many keypoints sit at exact identical (x, y) across DIFFERENT animals (static camera-trap background / watermark!)
print("\n=== 4. STATIC CAMERA-TRAP / WATERMARK KEYPOINT CHECK ===")
sift_z = np.load("cache_v2/sift_peak_pairs_csr.npz")
row_ptr, col_idx, feat_vals = sift_z["row_ptr"], sift_z["col_idx"], sift_z["feat_vals"]
# Check false-positive SIFT matches on train.csv (different identities with high sim_inl!)
tr_ids = train_df["identity"].values
tr_sp = train_df["species"].values
for sp in ["hyena", "leopard"]:
    idx_sp = np.where(tr_sp == sp)[0]
    same_inl, diff_inl = [], []
    diff_high_lowcos = 0
    diff_high_all = 0
    same_high_all = 0
    for i in idx_sp:
        s, e = row_ptr[i], row_ptr[i + 1]
        nbrs = col_idx[s:e]
        fv = feat_vals[s:e]
        m_tr = (nbrs < n_tr) & (nbrs > i)
        if not np.any(m_tr):
            continue
        nb = nbrs[m_tr]
        sim_i = fv[m_tr, 4]
        is_same = (tr_ids[nb] == tr_ids[i])
        if np.any(is_same):
            same_inl.append(sim_i[is_same])
            same_high_all += int((sim_i[is_same] >= 12).sum())
        if np.any(~is_same):
            d_sim = sim_i[~is_same]
            diff_inl.append(d_sim)
            diff_high_all += int((d_sim >= 12).sum())
            cos_d = X_tr[i] @ X_tr[nb[~is_same]].T
            diff_high_lowcos += int(((d_sim >= 12) & (cos_d < 0.35)).sum())
    si = np.concatenate(same_inl) if same_inl else np.array([0])
    di = np.concatenate(diff_inl) if diff_inl else np.array([0])
    print(
        f"  [{sp:7s}] Train pairs SIFT sim_inl: SAME med={np.median(si):.1f}, p90={np.quantile(si,0.9):.1f}, >=12 count={same_high_all} | "
        f"DIFF med={np.median(di):.1f}, p99={np.quantile(di,0.99):.1f}, >=12 count={diff_high_all} (of which cos<0.35: {diff_high_lowcos})"
    )

# 5. CRITICAL TRANSDUCTIVE TEST-CLUSTER CONSISTENCY AUDIT OF OUR 0.854 SUBMISSION!
print("\n=== 5. TEST-CLUSTER CONSISTENCY & BORDERLINE AUDIT OF 0.854 SUBMISSION ===")
for sp in ["hyena", "leopard"]:
    with open(f"cache_v2/v2_bundles_{sp}.pkl", "rb") as f:
        d = pickle.load(f)
    tb = d["test_bundle"]
    m_sp = (test_df["species"].values == sp)
    r1 = sub_854["predictions"].str.split(" ").str[0].values[m_sp]
    r2 = sub_854["predictions"].str.split(" ").str[1].values[m_sp]
    z60 = tb["X_open_base"][:, 55]  # z60 index check via raw_s1 too
    raw_s1 = tb["raw_s1"]
    sim_inl = tb["max_sim_inl"]

    # Check high-confidence query-to-query pairs (either strong W_qq OR strong Q-Q SIFT!)
    W_qq = tb["W_qq"]
    # Also find Q-Q SIFT inliers directly from CSR!
    te_glob = np.where(m_sp)[0] + n_tr
    g2l = np.full(len(miew_all), -1, dtype=np.int32)
    g2l[te_glob] = np.arange(len(te_glob), dtype=np.int32)
    qq_sift_adj = np.zeros((len(te_glob), len(te_glob)), dtype=bool)
    qq_sift_inl_cnt = 0
    for li, gi in enumerate(te_glob):
        s, e = row_ptr[gi], row_ptr[gi + 1]
        nbrs = col_idx[s:e]
        fv = feat_vals[s:e]
        ql = g2l[nbrs]
        qm = (ql >= 0) & (ql != li) & (fv[:, 4] >= 12)
        if np.any(qm):
            qq_sift_adj[li, ql[qm]] = True
            qq_sift_inl_cnt += int(qm.sum())

    # Connected components by strong MiewID (W_qq > 0.50^3.5) OR SIFT>=12!
    strong_adj = (W_qq >= (0.48 ** 3.5)) | qq_sift_adj
    nc_s, comp_s = connected_components(strong_adj, directed=False)
    n_mixed_known_new = 0
    n_mixed_rows = 0
    n_mixed_diff_ids = 0
    for c in range(nc_s):
        idx = np.where(comp_s == c)[0]
        if len(idx) < 2:
            continue
        preds_c = r1[idx]
        has_new = np.any(preds_c == "new_individual")
        kn_ids = set(preds_c[preds_c != "new_individual"])
        if has_new and len(kn_ids) >= 1:
            n_mixed_known_new += 1
            n_mixed_rows += len(idx)
        if len(kn_ids) >= 2:
            n_mixed_diff_ids += 1

    print(
        f"  [{sp:7s}] Strong test clusters (sim>=0.48 or QQ-SIFT>=12): "
        f"n_clusters_ge2={sum(np.bincount(comp_s)>=2)}, qq_sift>=12_edges={qq_sift_inl_cnt//2} | "
        f"CONFLICTS inside same-animal clusters: mixed(known+new)={n_mixed_known_new} clusters ({n_mixed_rows} images!), mixed(2+ diff known IDs)={n_mixed_diff_ids} clusters!!"
    )

    # Check borderline rows in 0.854 submission!
    is_new_pred = (r1 == "new_individual")
    print(
        f"  [{sp:7s}] 0.854 Borderline audit: "
        f"Pred=NEW with raw_s1>=0.45: {(is_new_pred & (raw_s1>=0.45)).sum()}, with sim_inl>=12 & raw_s1>=0.38: {(is_new_pred & (sim_inl>=12) & (raw_s1>=0.38)).sum()} | "
        f"Pred=KNOWN with raw_s1<0.32 & sim_inl<6: {((~is_new_pred) & (raw_s1<0.32) & (sim_inl<6)).sum()}"
    )

# 6. EXACT ERROR ANATOMY ON 5-FOLD OUTER CV (WHERE ARE WE LOSING EVERY 0.001 OF MAP@5?)
print("\n=== 6. EXACT CV ERROR ANATOMY (CROSS-VIEW vs SAME-VIEW, RARE vs FREQ, TOP-12 MISSES) ===")
from cv_v2 import build_or_load_cv_v2
folds = build_or_load_cv_v2()
for sp in ["hyena", "leopard"]:
    with open(f"cache_v2/v2_bundles_{sp}.pkl", "rb") as f:
        d = pickle.load(f)
    # Inspect Stage-1 recall & cross-view difficulty across all 5 outer folds!
    tot_kn, miss12_kn, miss1_kn = 0, 0, 0
    cv_same_vp_hit1, cv_same_vp_n = 0, 0
    cv_diff_vp_hit1, cv_diff_vp_n = 0, 0
    sing_hit1, sing_rec12, sing_n = 0, 0, 0
    small_hit1, small_rec12, small_n = 0, 0, 0
    freq_hit1, freq_rec12, freq_n = 0, 0, 0

    for fk in range(5):
        fd = folds[fk]
        gi = fd["gallery_idx"]
        g_sp = gi[train_df["species"].values[gi] == sp]
        g_ids = train_df["identity"].values[g_sp]
        g_vps = train_df["viewpoint"].fillna("MISSING").values[g_sp]
        vc = pd.Series(g_ids).value_counts()
        vp_by_id = {}
        for gid, gvp in zip(g_ids, g_vps):
            vp_by_id.setdefault(gid, set()).add(gvp)

        vb = d["outer_data"][fk]["val_bundle"]
        v_sp = fd["val_idx"][train_df["species"].values[fd["val_idx"]] == sp]
        v_vps = train_df["viewpoint"].fillna("MISSING").values[v_sp]
        kn = ~vb["is_new"]
        for i in np.where(kn)[0]:
            yt = vb["y_true"][i]
            cands = vb["cand_names"][i]
            h1 = (cands[0] == yt)
            h12 = (yt in cands)
            tot_kn += 1
            miss1_kn += int(not h1)
            miss12_kn += int(not h12)
            cnt = int(vc.get(yt, 0))
            if cnt == 1:
                sing_n += 1; sing_hit1 += int(h1); sing_rec12 += int(h12)
            elif cnt <= 4:
                small_n += 1; small_hit1 += int(h1); small_rec12 += int(h12)
            else:
                freq_n += 1; freq_hit1 += int(h1); freq_rec12 += int(h12)
            if v_vps[i] in vp_by_id.get(yt, set()):
                cv_same_vp_n += 1; cv_same_vp_hit1 += int(h1)
            else:
                cv_diff_vp_n += 1; cv_diff_vp_hit1 += int(h1)

    print(f"  [{sp.upper()}] Total Known Val={tot_kn} | Stage-1 Top-1={1-miss1_kn/tot_kn:.4f} | Top-12 Rec={1-miss12_kn/tot_kn:.4f}")
    print(f"     Same-VP in gallery (n={cv_same_vp_n:4d}, {cv_same_vp_n/tot_kn:.1%}): Stage-1 Top-1 = {cv_same_vp_hit1/max(1,cv_same_vp_n):.4f}")
    print(f"     CROSS-VIEW ONLY    (n={cv_diff_vp_n:4d}, {cv_diff_vp_n/tot_kn:.1%}): Stage-1 Top-1 = {cv_diff_vp_hit1/max(1,cv_diff_vp_n):.4f}")
    print(f"     Gallery n=1 (sing) (n={sing_n:4d}): Top-1={sing_hit1/max(1,sing_n):.4f}, Top-12={sing_rec12/max(1,sing_n):.4f}")
    print(f"     Gallery n=2..4     (n={small_n:4d}): Top-1={small_hit1/max(1,small_n):.4f}, Top-12={small_rec12/max(1,small_n):.4f}")
    print(f"     Gallery n>=5       (n={freq_n:4d}): Top-1={freq_hit1/max(1,freq_n):.4f}, Top-12={freq_rec12/max(1,freq_n):.4f}")
