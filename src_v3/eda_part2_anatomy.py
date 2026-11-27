import sys
sys.path.insert(0, "src_v2")
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import connected_components
from scipy.stats import gaussian_kde
from sklearn.preprocessing import normalize
from cv_v2 import build_or_load_cv_v2

train_df = pd.read_csv("reid_student_data_main/train.csv")
test_df = pd.read_csv("reid_student_data_main/test.csv")
sub_854 = pd.read_csv("submissions_v2/submission_0854_verified_backup.csv")
folds = build_or_load_cv_v2()

print("=== 6. EXACT CV ERROR ANATOMY (CROSS-VIEW vs SAME-VIEW, RARE vs FREQ, TOP-12 MISSES) ===")
for sp in ["hyena", "leopard"]:
    with open(f"cache_v2/v2_bundles_{sp}.pkl", "rb") as f:
        d = pickle.load(f)
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

print("\n=== 7. EXACT MULTI-FEATURE KDE MIXTURE MLE FOR TRUE TEST pi_new (LOIO-INVARIANT!) ===")
from diagnose_v2 import OPEN_NAMES
for sp in ["hyena", "leopard"]:
    with open(f"cache_v2/v2_bundles_{sp}.pkl", "rb") as f:
        d = pickle.load(f)
    Xcv = np.vstack([od["val_bundle"]["X_open_base"] for od in d["outer_data"]])
    is_new_cv = np.concatenate([od["val_bundle"]["is_new"] for od in d["outer_data"]])
    Xte = d["test_bundle"]["X_open_base"]

    print(f"\n  [{sp.upper()}] True test pi_new MLE across individual domain-invariant features:")
    for fn in ["zraw", "z60", "z45", "zlda", "ecraw", "ec60", "raw_g12", "p60_g12", "sift_sim_max"]:
        ci = OPEN_NAMES.index(fn)
        x_new = Xcv[is_new_cv, ci]
        x_kn = Xcv[~is_new_cv, ci]
        x_te = Xte[:, ci]
        kde_n = gaussian_kde(x_new + np.random.RandomState(42).normal(0, 1e-3, len(x_new)))
        kde_k = gaussian_kde(x_kn + np.random.RandomState(42).normal(0, 1e-3, len(x_kn)))
        pn = np.maximum(1e-12, kde_n(x_te))
        pk = np.maximum(1e-12, kde_k(x_te))
        res = minimize_scalar(lambda pi: -np.mean(np.log(pi * pn + (1.0 - pi) * pk)), bounds=(0.05, 0.95), method="bounded")
        print(f"     {fn:14s}: MLE pi_new = {res.x:.4f} (cv_new_med={np.median(x_new):+.3f}, cv_kn_med={np.median(x_kn):+.3f}, te_med={np.median(x_te):+.3f})")

    # 2D Joint KDE MLE on (z60, zraw) and (z60, ecraw)!
    for f1, f2 in [("z60", "zraw"), ("z60", "raw_g12"), ("zraw", "sift_sim_max")]:
        c1, c2 = OPEN_NAMES.index(f1), OPEN_NAMES.index(f2)
        rng = np.random.RandomState(42)
        vn = Xcv[is_new_cv][:, [c1, c2]].T + rng.normal(0, 1e-3, (2, int(is_new_cv.sum())))
        vk = Xcv[~is_new_cv][:, [c1, c2]].T + rng.normal(0, 1e-3, (2, int((~is_new_cv).sum())))
        vt = Xte[:, [c1, c2]].T
        pn = np.maximum(1e-12, gaussian_kde(vn)(vt))
        pk = np.maximum(1e-12, gaussian_kde(vk)(vt))
        res = minimize_scalar(lambda pi: -np.mean(np.log(pi * pn + (1.0 - pi) * pk)), bounds=(0.05, 0.95), method="bounded")
        print(f"     2D ({f1:6s}, {f2:12s}): MLE pi_new = {res.x:.4f}")

print("\n=== 8. INSPECT SAMPLE OF 0.854 MISSED HIGH-SIFT LEOPARDS & MIXED CLUSTERS ===")
with open("cache_v2/v2_bundles_leopard.pkl", "rb") as f:
    dl = pickle.load(f)
tbl = dl["test_bundle"]
ml = (test_df["species"].values == "leopard")
te_ids = test_df["id"].values[ml]
te_vps = test_df["viewpoint"].fillna("MISSING").values[ml]
r1 = sub_854["predictions"].str.split(" ").str[0].values[ml]
r2 = sub_854["predictions"].str.split(" ").str[1].values[ml]
z60 = tbl["X_open_base"][:, OPEN_NAMES.index("z60")]
raw_s1 = tbl["raw_s1"]
sim_inl = tbl["max_sim_inl"]
vp_frac0 = tbl["X_open_base"][:, OPEN_NAMES.index("vp_frac0")]

miss_idx = np.where((r1 == "new_individual") & (sim_inl >= 12) & (raw_s1 >= 0.38))[0]
print(f"Leopard 0.854 Pred=NEW with sim_inl>=12 & raw_s1>=0.38 (n={len(miss_idx)}, first 15):")
for li in miss_idx[:15]:
    print(f"  {te_ids[li]} (vp={te_vps[li]:7s}) | R2={r2[li]:8s} | raw_s1={raw_s1[li]:.3f} | z60={z60[li]:+.2f} | sim_inl={sim_inl[li]:3.0f} | vp_frac0={vp_frac0[li]:.2f}")
