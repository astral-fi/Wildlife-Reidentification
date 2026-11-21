import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components
from scipy.optimize import minimize
from scipy import stats

DATA_DIR = "reid_student_data_main"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
miew_feats = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))

n_tr = len(train_df)
X_norm = normalize(miew_feats, axis=1)
X_tr = X_norm[:n_tr]
X_te = X_norm[n_tr:]

print("=== 1. CHECK ROW ORDERING / ADJACENT SIMILARITY ===")
adj_tr_sim = np.sum(X_tr[:-1] * X_tr[1:], axis=1)
adj_te_sim = np.sum(X_te[:-1] * X_te[1:], axis=1)
same_sp_tr = (train_df["species"].values[:-1] == train_df["species"].values[1:])
same_id_tr = (train_df["identity"].values[:-1] == train_df["identity"].values[1:])
print(f"Train adjacent same-species rate: {same_sp_tr.mean():.4f}, adjacent same-ID rate: {same_id_tr.mean():.4f}, mean adj sim: {adj_tr_sim.mean():.4f}")
same_sp_te = (test_df["species"].values[:-1] == test_df["species"].values[1:])
print(f"Test  adjacent same-species rate: {same_sp_te.mean():.4f}, mean adj sim: {adj_te_sim.mean():.4f}")

bins = [0.2, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]

for sp in ["hyena", "leopard"]:
    print(f"\n================ {sp.upper()} DEEP SIMILARITY & SPLIT STRUCTURE ================")
    tr_idx = np.where(train_df["species"].values == sp)[0]
    te_idx = np.where(test_df["species"].values == sp)[0]
    Xt = X_tr[tr_idx]
    Xq = X_te[te_idx]
    ids_t = train_df["identity"].values[tr_idx]
    vps_t = train_df["viewpoint"].fillna("MISSING").values[tr_idx]
    vps_q = test_df["viewpoint"].fillna("MISSING").values[te_idx]

    S_tt = Xt @ Xt.T
    np.fill_diagonal(S_tt, -2.0)
    S_qt = Xq @ Xt.T
    S_qq = Xq @ Xq.T
    np.fill_diagonal(S_qq, -2.0)

    # Different-ID pairwise similarity baseline
    same_id_mat = (ids_t[:, None] == ids_t[None, :])
    same_vp_mat = (vps_t[:, None] == vps_t[None, :])
    diff_same_vp = S_tt[(~same_id_mat) & same_vp_mat]
    diff_diff_vp = S_tt[(~same_id_mat) & (~same_vp_mat)]
    print(f"Impostor (Diff-ID) Pairwise Sim: Same-VP mean={diff_same_vp.mean():.3f} (q99={np.quantile(diff_same_vp,0.99):.3f}), Diff-VP mean={diff_diff_vp.mean():.3f} (q99={np.quantile(diff_diff_vp,0.99):.3f})")

    # Burst analysis within each train ID at sim >= 0.85
    burst_adj = (S_tt >= 0.85) & same_id_mat
    n_comp, comp_labels = connected_components(burst_adj, directed=False)
    print(f"Train rows={len(tr_idx)}, unique IDs={len(np.unique(ids_t))}, sim>=0.85 burst clusters={n_comp}")

    # Compare 3 reference max-sim distributions:
    # 1) LOO max-sim (for IDs with >=2 images)
    # 2) Leave-One-Burst-Out (LOBO) max-sim (masking same burst_cluster, for IDs with >=2 bursts)
    # 3) Leave-One-ID-Out (LOIO) max-sim (simulating true new_individual)
    id_counts = pd.Series(ids_t).value_counts()
    id_burst_counts = pd.DataFrame({"id": ids_t, "b": comp_labels}).groupby("id")["b"].nunique()

    loo_known, lobo_known, loio_new = [], [], []
    for i in range(len(tr_idx)):
        my_id = ids_t[i]
        loio_new.append(S_tt[i, ids_t != my_id].max())
        if id_counts[my_id] >= 2:
            loo_known.append(S_tt[i].max())
        if id_burst_counts[my_id] >= 2:
            mask_not_my_burst = (comp_labels != comp_labels[i])
            lobo_known.append(S_tt[i, mask_not_my_burst].max())

    loo_known = np.array(loo_known)
    lobo_known = np.array(lobo_known)
    loio_new = np.array(loio_new)
    test_max = S_qt.max(axis=1)

    h_loo, _ = np.histogram(loo_known, bins=bins, density=False)
    h_lobo, _ = np.histogram(lobo_known, bins=bins, density=False)
    h_new, _ = np.histogram(loio_new, bins=bins, density=False)
    h_te, _ = np.histogram(test_max, bins=bins, density=False)

    df_hist = pd.DataFrame({
        "bin": [f"[{bins[k]:.2f},{bins[k+1]:.2f})" for k in range(len(bins)-1)],
        "LOO_known%": (100 * h_loo / len(loo_known)).round(1),
        "LOBO_known%": (100 * h_lobo / len(lobo_known)).round(1),
        "Unseen_new%": (100 * h_new / len(loio_new)).round(1),
        "TEST%": (100 * h_te / len(test_max)).round(1),
        "TEST_n": h_te
    })
    print("\nMax-Gallery Cosine Similarity Histograms:")
    print(df_hist.to_string(index=False))

    # Fit mixture under BOTH LOO_known and LOBO_known!
    for name, ref_k in [("LOO (random row split)", loo_known), ("LOBO (encounter/time split)", lobo_known)]:
        kde_k = stats.gaussian_kde(ref_k, bw_method=0.15)
        kde_n = stats.gaussian_kde(loio_new, bw_method=0.15)
        pk, pn = kde_k(test_max), kde_n(test_max)
        res = minimize(lambda p: -np.sum(np.log(np.clip((1 - p[0]) * pk + p[0] * pn, 1e-12, None))),
                       x0=[0.35], bounds=[(0.0, 1.0)])
        ll = -res.fun
        print(f"Mixture fit [{name}]: pi_new = {res.x[0]:.4f} (logLik={ll:.1f})")

    # Inspect Test-to-Test vs Test-to-Train joint behavior!
    tt_max = S_qq.max(axis=1)
    high_tt_low_qt = ((tt_max > 0.75) & (test_max < 0.52)).sum()
    high_tt_high_qt = ((tt_max > 0.75) & (test_max >= 0.65)).sum()
    print(f"Test rows with Test-to-Test sim > 0.75: {(tt_max > 0.75).sum()}/{len(te_idx)} | of which Test-to-Train < 0.52 (unseen test clusters!): {high_tt_low_qt}, >=0.65 (known test clusters!): {high_tt_high_qt}")
