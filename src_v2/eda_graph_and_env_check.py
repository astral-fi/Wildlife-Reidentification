import os
import importlib.util
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize

print("optuna installed:", importlib.util.find_spec("optuna") is not None)

train_df = pd.read_csv("reid_student_data_main/train.csv")
test_df = pd.read_csv("reid_student_data_main/test.csv")
miew = np.load("reid_student_data_main/miewid_features.npy")
n_tr, n_te = len(train_df), len(test_df)

for sp in ["hyena", "leopard"]:
    tr_i = np.where(train_df["species"].values == sp)[0]
    te_i = n_tr + np.where(test_df["species"].values == sp)[0]
    all_i = np.concatenate([tr_i, te_i])
    is_te = np.array([False] * len(tr_i) + [True] * len(te_i))

    # SignedPower 0.85 + PCA Whitening p=0.55 (top 640 / 768)
    X = np.sign(miew[all_i]) * (np.abs(miew[all_i]) ** 0.85)
    X = normalize(X, axis=1).astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / len(Xc)
    ev, V = np.linalg.eigh(C)
    d = 640 if sp == "hyena" else 768
    Z = normalize((Xc @ V[:, -d:]) * ((ev[-d:] + 1.5e-3) ** (-0.275))[None, :], axis=1).astype(np.float32)

    S = Z @ Z.T
    np.fill_diagonal(S, -2.0)

    # For test queries: max sim to TRAIN vs max sim to TEST, and fraction of top-K neighbors in TEST!
    S_qt = S[is_te][:, ~is_te]
    S_qq = S[is_te][:, is_te]
    max_tr = S_qt.max(axis=1)
    max_te = S_qq.max(axis=1)

    top10_idx = np.argsort(-S[is_te], axis=1)[:, :10]
    top10_te_frac = is_te[top10_idx].mean(axis=1)
    top5_te_frac = is_te[top10_idx[:, :5]].mean(axis=1)

    base_te_share = len(te_i) / len(all_i)
    # Test Adv AUC ONLY on high-train-match test rows (known test animals!) vs train!
    high_tr_mask = max_tr >= np.quantile(max_tr, 0.70)
    X_known_adv = np.vstack([Z[~is_te], Z[is_te][high_tr_mask]])
    y_known_adv = np.array([0] * len(tr_i) + [1] * int(high_tr_mask.sum()))
    auc_known_only = cross_val_score(LogisticRegression(C=1.0, max_iter=300), X_known_adv, y_known_adv, cv=5, scoring="roc_auc").mean()

    # Check strong-test-only clusters (max_te >= 0.42 AND max_tr < 0.30 in whitened space!)
    strong_te_no_tr = ((max_te >= 0.40) & (max_tr < 0.28)).sum()
    strong_te_yes_tr = ((max_te >= 0.40) & (max_tr >= 0.35)).sum()

    # Connected components in TEST at whitened sim >= 0.38
    nc, comp = connected_components(S_qq >= 0.38, directed=False)
    comp_sizes = pd.Series(comp).value_counts()
    n_multi_comp = (comp_sizes >= 2).sum()
    rows_in_multi = comp_sizes[comp_sizes >= 2].sum()

    # Check how many rows in multi-image test clusters have HUGE gap between their own max_tr and their cluster's max_tr!
    rescue_candidates = 0
    pure_new_cluster_rows = 0
    for cid, sz in comp_sizes.items():
        if sz >= 2:
            m_c = (comp == cid)
            c_max = max_tr[m_c].max()
            if c_max >= 0.42:
                rescue_candidates += int((max_tr[m_c] < 0.30).sum())
            elif c_max < 0.30:
                pure_new_cluster_rows += int(sz)

    print(f"\n================ {sp.upper()} JOINT TRAIN+TEST GRAPH DISCOVERY ================")
    print(f"  Base test share in (train+test): {base_te_share:.3f}")
    print(f"  Adv AUC (Known-only top-30% test vs train): {auc_known_only:.4f}")
    print(f"  Test queries top-5 neighbors in TEST: mean={top5_te_frac.mean():.3f} | frac with >=80% top-5 in TEST: {(top5_te_frac >= 0.80).mean():.3f}")
    print(f"  Test queries with max_te>=0.40 & max_tr<0.28 (clear multi-frame UNSEEN!): {strong_te_no_tr}/{len(te_i)} ({strong_te_no_tr/len(te_i):.1%})")
    print(f"  Test queries with max_te>=0.40 & max_tr>=0.35 (clear multi-frame KNOWN!): {strong_te_yes_tr}/{len(te_i)} ({strong_te_yes_tr/len(te_i):.1%})")
    print(f"  Test clusters (whiten>=0.38): {n_multi_comp} multi-image clusters covering {rows_in_multi}/{len(te_i)} ({rows_in_multi/len(te_i):.1%}) test rows!")
    print(f"    -> Hard known test queries RESCUABLE by a strong sibling in their cluster: {rescue_candidates}")
    print(f"    -> Confirmed UNSEEN test queries inside pure-unseen multi-frame clusters: {pure_new_cluster_rows}")
