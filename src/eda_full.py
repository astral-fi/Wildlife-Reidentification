import os
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import normalize

t0 = time.time()
DATA_DIR = "reid_student_data_main"
SIFT_TR_DIR = "sift_features_train.npz"
SIFT_TE_DIR = "sift_features_test.npz"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
miew_ids = pd.read_csv(os.path.join(DATA_DIR, "miewid_ids.csv"))
miew_feats = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))

print("=== 1. INTEGRITY & ALIGNMENT ===")
print(f"train_df: {train_df.shape}, cols={list(train_df.columns)}")
print(f"test_df:  {test_df.shape}, cols={list(test_df.columns)}")
print(f"sub_df:   {sub_df.shape}, cols={list(sub_df.columns)}")
print(f"miew_ids: {miew_ids.shape}, miew_feats: {miew_feats.shape} ({miew_feats.dtype})")

expected_ids = pd.concat([train_df["id"], test_df["id"]], ignore_index=True)
exact_match = (miew_ids["id"] == expected_ids).all()
sub_match = (sub_df["id"] == test_df["id"]).all()
print(f"miew_ids exact match [train; test]: {exact_match}")
print(f"sample_submission exact match test: {sub_match}")
print(f"miew_feats NaNs: {np.isnan(miew_feats).sum()}, Infs: {np.isinf(miew_feats).sum()}")
print(f"train nulls:\n{train_df.isna().sum().to_dict()}")
print(f"test nulls:\n{test_df.isna().sum().to_dict()}")

# Check species-identity disjointness
id_species = train_df.groupby("identity")["species"].nunique()
print(f"Identities spanning >1 species: {(id_species > 1).sum()}")
print("Identity prefixes by species:")
for sp, grp in train_df.groupby("species"):
    prefixes = set(grp["identity"].str[:3])
    print(f"  {sp}: n_rows={len(grp)}, n_identities={grp['identity'].nunique()}, prefixes={prefixes}")

print("\n=== 2. IDENTITY FREQUENCY & GOOD-TURING ANALYSIS ===")
for sp in ["hyena", "leopard"]:
    vc = train_df[train_df["species"] == sp]["identity"].value_counts()
    n_total = vc.sum()
    n_ids = len(vc)
    n1 = (vc == 1).sum()
    n2 = (vc == 2).sum()
    n3_5 = ((vc >= 3) & (vc <= 5)).sum()
    n6_10 = ((vc >= 6) & (vc <= 10)).sum()
    n11_plus = (vc > 10).sum()
    print(f"[{sp.upper()}] rows={n_total}, unique_IDs={n_ids}, mean_imgs/ID={n_total/n_ids:.2f}, median={vc.median():.1f}, max={vc.max()}")
    print(f"  ID count buckets: 1:{n1} ({n1/n_ids:.1%}), 2:{n2} ({n2/n_ids:.1%}), 3-5:{n3_5} ({n3_5/n_ids:.1%}), 6-10:{n6_10} ({n6_10/n_ids:.1%}), >10:{n11_plus} ({n11_plus/n_ids:.1%})")
    print(f"  Good-Turing singleton row rate (N1 / N_rows): {n1 / n_total:.4f} ({n1}/{n_total})")

print("\n=== 3. VIEWPOINT DISTRIBUTION (TRAIN vs TEST) ===")
for sp in ["hyena", "leopard"]:
    tr_vp = train_df[train_df["species"] == sp]["viewpoint"].fillna("MISSING").value_counts(normalize=True)
    te_vp = test_df[test_df["species"] == sp]["viewpoint"].fillna("MISSING").value_counts(normalize=True)
    vp_comp = pd.DataFrame({"train_pct": (tr_vp * 100).round(2), "test_pct": (te_vp * 100).round(2),
                            "train_n": train_df[train_df["species"] == sp]["viewpoint"].fillna("MISSING").value_counts(),
                            "test_n": test_df[test_df["species"] == sp]["viewpoint"].fillna("MISSING").value_counts()}).fillna(0)
    print(f"\n[{sp.upper()}] Viewpoints:")
    print(vp_comp.to_string())

# Check how many identities have multiple viewpoints (especially left AND right)
for sp in ["hyena", "leopard"]:
    sub = train_df[train_df["species"] == sp]
    id_vps = sub.groupby("identity")["viewpoint"].apply(lambda s: set(s.dropna()))
    multi_vp = id_vps.apply(lambda s: len(s) > 1).sum()
    has_lr = id_vps.apply(lambda s: ("left" in s or "frontleft" in s or "backleft" in s) and ("right" in s or "frontright" in s or "backright" in s)).sum()
    print(f"[{sp}] Identities with >1 viewpoint: {multi_vp}/{len(id_vps)} ({multi_vp/len(id_vps):.1%}), with BOTH left & right sides: {has_lr}/{len(id_vps)} ({has_lr/len(id_vps):.1%})")

print("\n=== 4. MIEWID FEATURE PROPERTIES & SPECTRUM ===")
n_tr = len(train_df)
X_tr = miew_feats[:n_tr]
X_te = miew_feats[n_tr:]
norms_tr = np.linalg.norm(X_tr, axis=1)
norms_te = np.linalg.norm(X_te, axis=1)
col_stds = miew_feats.std(axis=0)
print(f"Value min={miew_feats.min():.4f}, max={miew_feats.max():.4f}, mean={miew_feats.mean():.4f}, frac_neg={(miew_feats < 0).mean():.4f}")
print(f"Dead/near-zero std columns (<1e-6): {(col_stds < 1e-6).sum()}, min col std={col_stds.min():.5f}, max col std={col_stds.max():.5f}")
for sp in ["hyena", "leopard"]:
    ntr = norms_tr[train_df["species"].values == sp]
    nte = norms_te[test_df["species"].values == sp]
    print(f"[{sp}] L2 norm train: mean={ntr.mean():.2f}, std={ntr.std():.2f}, [min={ntr.min():.2f}, max={ntr.max():.2f}] | test: mean={nte.mean():.2f}, std={nte.std():.2f}")

X_norm = normalize(miew_feats, axis=1)
X_tr_norm = X_norm[:n_tr]
X_te_norm = X_norm[n_tr:]

for sp in ["hyena", "leopard"]:
    mask_all = np.concatenate([train_df["species"].values == sp, test_df["species"].values == sp])
    pca = PCA(n_components=min(1000, mask_all.sum() - 1), random_state=42)
    pca.fit(X_norm[mask_all])
    cvar = np.cumsum(pca.explained_variance_ratio_)
    d50 = np.searchsorted(cvar, 0.50) + 1
    d80 = np.searchsorted(cvar, 0.80) + 1
    d90 = np.searchsorted(cvar, 0.90) + 1
    d95 = np.searchsorted(cvar, 0.95) + 1
    d99 = np.searchsorted(cvar, 0.99) + 1
    print(f"[{sp}] PCA cumulative variance dims (on L2-normed train+test): 50%={d50}, 80%={d80}, 90%={d90}, 95%={d95}, 99%={d99}")

# Can miewid predict species & viewpoint? And Adversarial validation (train vs test)!
clf_sp = LogisticRegression(C=1.0, max_iter=500)
sp_acc = cross_val_score(clf_sp, X_tr_norm, (train_df["species"] == "leopard").astype(int), cv=5).mean()
print(f"Species 5-fold CV accuracy from miewid: {sp_acc:.5f}")

for sp in ["hyena", "leopard"]:
    # Adversarial validation: train (0) vs test (1)
    X_adv = np.vstack([X_tr_norm[train_df["species"].values == sp], X_te_norm[test_df["species"].values == sp]])
    y_adv = np.array([0] * (train_df["species"].values == sp).sum() + [1] * (test_df["species"].values == sp).sum())
    auc = cross_val_score(LogisticRegression(C=1.0, max_iter=500), X_adv, y_adv, cv=5, scoring="roc_auc").mean()
    # Major viewpoint predictability (left/right/front/back)
    vp_mask = (train_df["species"] == sp) & (train_df["viewpoint"].isin(["left", "right", "front", "back"]))
    vp_acc = cross_val_score(LogisticRegression(C=1.0, max_iter=500), X_tr_norm[vp_mask], train_df.loc[vp_mask, "viewpoint"], cv=5).mean()
    print(f"[{sp}] Adversarial AUC (train vs test): {auc:.4f} | 4-way Viewpoint CV accuracy: {vp_acc:.4f}")

print("\n=== 5. SIMILARITY STRUCTURE & MIXTURE MODEL ESTIMATION OF NEW_INDIVIDUAL RATE ===")
for sp in ["hyena", "leopard"]:
    tr_idx = np.where(train_df["species"].values == sp)[0]
    te_idx = np.where(test_df["species"].values == sp)[0]
    Xt = X_tr_norm[tr_idx]
    Xq = X_te_norm[te_idx]
    ids_t = train_df["identity"].values[tr_idx]
    vps_t = train_df["viewpoint"].fillna("MISSING").values[tr_idx]

    S_tt = Xt @ Xt.T
    np.fill_diagonal(S_tt, -2.0)

    # For each train sample with count >= 2: max sim to ANY other train sample (realistic known query)
    # and max sim to SAME identity vs DIFF identity
    id_counts = pd.Series(ids_t).value_counts()
    known_max_sim = []
    same_id_same_vp = []
    same_id_diff_vp = []
    same_id_opp_vp = []
    new_max_sim = []  # Mask out ALL samples of the same identity! Simulates true new_individual!

    opp_pairs = {("left", "right"), ("right", "left"), ("frontleft", "frontright"), ("frontright", "frontleft"), ("backleft", "backright"), ("backright", "backleft")}
    for i in range(len(tr_idx)):
        same_mask = (ids_t == ids_t[i])
        same_mask[i] = False
        diff_mask = (ids_t != ids_t[i])
        # Simulate this image arriving when its entire identity is NOT in gallery
        new_max_sim.append(S_tt[i, diff_mask].max())
        if same_mask.sum() >= 1:
            known_max_sim.append(S_tt[i].max())
            for j in np.where(same_mask)[0]:
                if j > i:
                    sim_ij = S_tt[i, j]
                    if vps_t[i] == vps_t[j]:
                        same_id_same_vp.append(sim_ij)
                    else:
                        same_id_diff_vp.append(sim_ij)
                        if (vps_t[i], vps_t[j]) in opp_pairs:
                            same_id_opp_vp.append(sim_ij)

    known_max_sim = np.array(known_max_sim)
    new_max_sim = np.array(new_max_sim)

    S_qt = Xq @ Xt.T
    test_max_sim = S_qt.max(axis=1)

    # Estimate pi_new via histogram / KDE log-likelihood mixture fit:
    # p_test(s) = (1 - pi) * p_known(s) + pi * p_new(s)
    kde_known = stats.gaussian_kde(known_max_sim, bw_method=0.15)
    kde_new = stats.gaussian_kde(new_max_sim, bw_method=0.15)
    pk = kde_known(test_max_sim)
    pn = kde_new(test_max_sim)
    res = minimize(lambda p: -np.sum(np.log(np.clip((1 - p[0]) * pk + p[0] * pn, 1e-12, None))),
                   x0=[0.25], bounds=[(0.0, 1.0)])
    pi_est = res.x[0]

    # Also check top-1 raw kNN LOO accuracy on seen individuals (count >= 2)
    seen_correct_top1 = 0
    seen_correct_top5 = 0
    seen_n = 0
    for i in range(len(tr_idx)):
        if id_counts[ids_t[i]] >= 2:
            seen_n += 1
            order = np.argsort(-S_tt[i])
            ranked_ids = []
            for idx_j in order:
                cand = ids_t[idx_j]
                if cand not in ranked_ids:
                    ranked_ids.append(cand)
                    if len(ranked_ids) == 5:
                        break
            if ranked_ids[0] == ids_t[i]:
                seen_correct_top1 += 1
            if ids_t[i] in ranked_ids:
                seen_correct_top5 += 1

    print(f"\n[{sp.upper()}] Similarity & Open-Set Structure:")
    print(f"  Pairwise Same-ID Same-VP mean={np.mean(same_id_same_vp):.3f} (median={np.median(same_id_same_vp):.3f}, n={len(same_id_same_vp)})")
    print(f"  Pairwise Same-ID Diff-VP mean={np.mean(same_id_diff_vp):.3f} (median={np.median(same_id_diff_vp):.3f}, n={len(same_id_diff_vp)})")
    print(f"  Pairwise Same-ID Left-vs-Right mean={np.mean(same_id_opp_vp):.3f} (median={np.median(same_id_opp_vp):.3f}, n={len(same_id_opp_vp)})")
    print(f"  Max-Gallery-Sim for KNOWN IDs: mean={known_max_sim.mean():.3f}, q10={np.quantile(known_max_sim,0.1):.3f}, median={np.median(known_max_sim):.3f}, q90={np.quantile(known_max_sim,0.9):.3f}")
    print(f"  Max-Gallery-Sim for UNSEEN IDs: mean={new_max_sim.mean():.3f}, q10={np.quantile(new_max_sim,0.1):.3f}, median={np.median(new_max_sim):.3f}, q90={np.quantile(new_max_sim,0.9):.3f}")
    print(f"  Max-Gallery-Sim in TEST:        mean={test_max_sim.mean():.3f}, q10={np.quantile(test_max_sim,0.1):.3f}, median={np.median(test_max_sim):.3f}, q90={np.quantile(test_max_sim,0.9):.3f}")
    print(f"  >>> KDE Mixture Estimated test `new_individual` fraction pi_new = {pi_est:.4f} (~{int(round(pi_est * len(te_idx)))}/{len(te_idx)} test rows) <<<")
    print(f"  Raw L2-cosine LOO retrieval on seen train IDs (count>=2, n={seen_n}): Top1={seen_correct_top1/seen_n:.4f}, Top5={seen_correct_top5/seen_n:.4f}")

    # Check Test-to-Test nearest neighbor sim (are there clusters within test?)
    S_qq = Xq @ Xq.T
    np.fill_diagonal(S_qq, -2.0)
    tt_max = S_qq.max(axis=1)
    print(f"  Test-to-Test max sim: mean={tt_max.mean():.3f}, frac > 0.70: {(tt_max > 0.70).mean():.3f}, frac > 0.80: {(tt_max > 0.80).mean():.3f}")

print("\n=== 6. FULL SIFT DATASET SCAN (ALL 9,954 IMAGES) ===")
def scan_sift(ids, folder):
    k_counts = np.zeros(len(ids), dtype=np.int32)
    max_x = np.full(len(ids), np.nan, dtype=np.float32)
    max_y = np.full(len(ids), np.nan, dtype=np.float32)
    min_x = np.full(len(ids), np.nan, dtype=np.float32)
    min_y = np.full(len(ids), np.nan, dtype=np.float32)
    missing_files = 0
    bad_dtypes = 0
    for idx, img_id in enumerate(ids):
        dp = os.path.join(folder, f"{img_id}_desc.npy")
        xp = os.path.join(folder, f"{img_id}_xy.npy")
        if not (os.path.exists(dp) and os.path.exists(xp)):
            missing_files += 1
            continue
        desc = np.load(dp, mmap_mode="r")
        xy = np.load(xp, mmap_mode="r")
        k = desc.shape[0]
        k_counts[idx] = k
        if desc.dtype != np.uint8 or xy.dtype != np.float32 or (k > 0 and (desc.shape[1] != 128 or xy.shape != (k, 2))):
            bad_dtypes += 1
        if k > 0:
            xy_arr = np.asarray(xy)
            max_x[idx] = xy_arr[:, 0].max()
            max_y[idx] = xy_arr[:, 1].max()
            min_x[idx] = xy_arr[:, 0].min()
            min_y[idx] = xy_arr[:, 1].min()
    return k_counts, min_x, max_x, min_y, max_y, missing_files, bad_dtypes

k_tr, minx_tr, maxx_tr, miny_tr, maxy_tr, miss_tr, bad_tr = scan_sift(train_df["id"].values, SIFT_TR_DIR)
k_te, minx_te, maxx_te, miny_te, maxy_te, miss_te, bad_te = scan_sift(test_df["id"].values, SIFT_TE_DIR)

print(f"Train SIFT missing={miss_tr}, bad_shapes={bad_tr} | Test SIFT missing={miss_te}, bad_shapes={bad_te}")
for sp in ["hyena", "leopard"]:
    ktr_s = k_tr[train_df["species"].values == sp]
    kte_s = k_te[test_df["species"].values == sp]
    mx_tr = maxx_tr[train_df["species"].values == sp]
    my_tr = maxy_tr[train_df["species"].values == sp]
    print(f"[{sp}] Train K: zero={(ktr_s==0).sum()}, <50={(ktr_s<50).sum()}, ==500={(ktr_s==500).sum()}/{len(ktr_s)} ({(ktr_s==500).mean():.1%}), mean={ktr_s.mean():.1f}, min={ktr_s.min()}")
    print(f"[{sp}] Test  K: zero={(kte_s==0).sum()}, <50={(kte_s<50).sum()}, ==500={(kte_s==500).sum()}/{len(kte_s)} ({(kte_s==500).mean():.1%}), mean={kte_s.mean():.1f}, min={kte_s.min()}")
    print(f"[{sp}] XY max coords (train): max_x median={np.nanmedian(mx_tr):.1f} [q05={np.nanquantile(mx_tr,0.05):.1f}, q95={np.nanquantile(mx_tr,0.95):.1f}], max_y median={np.nanmedian(my_tr):.1f} [q05={np.nanquantile(my_tr,0.05):.1f}, q95={np.nanquantile(my_tr,0.95):.1f}]")

print(f"\nDone full EDA in {time.time() - t0:.1f}s")
