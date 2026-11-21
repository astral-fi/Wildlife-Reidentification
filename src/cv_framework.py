import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components

DATA_DIR = "reid_student_data_main"
CACHE_DIR = "cache"

# Exact test.csv species shares & empirically estimated new_individual priors (LOBO KDE mixture)
TEST_SPECIES_WEIGHTS = {
    "hyena": 1217.0 / 2791.0,
    "leopard": 1574.0 / 2791.0,
}
EST_PI_NEW = {
    "hyena": 0.395,
    "leopard": 0.605,
}


def ap_at_5(y_true_str, pred_list):
    seen = set()
    rank = 0
    for p in pred_list:
        if p in seen:
            continue
        seen.add(p)
        rank += 1
        if p == y_true_str:
            return 1.0 / rank
        if rank == 5:
            break
    return 0.0


def map_at_5(y_true, y_preds, sample_weight=None):
    aps = np.array([ap_at_5(yt, yp) for yt, yp in zip(y_true, y_preds)], dtype=np.float64)
    if sample_weight is None:
        return float(aps.mean())
    sw = np.asarray(sample_weight, dtype=np.float64)
    return float(np.sum(aps * sw) / np.sum(sw))


def build_or_load_cv(n_splits=5, burst_thresh=0.875, seed=42, force_rebuild=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"cv_folds_k{n_splits}_b{int(burst_thresh*1000)}_s{seed}.json")
    if os.path.exists(cache_path) and not force_rebuild:
        with open(cache_path, "r") as f:
            data = json.load(f)
        for fold in data["folds"]:
            fold["gallery_idx"] = np.array(fold["gallery_idx"], dtype=np.int32)
            fold["val_idx"] = np.array(fold["val_idx"], dtype=np.int32)
            fold["val_is_new"] = np.array(fold["val_is_new"], dtype=bool)
        return data

    rng = np.random.RandomState(seed)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    miew_feats = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))[:len(train_df)]
    X_norm = normalize(miew_feats, axis=1)

    # Compute burst clusters within each identity
    burst_id = np.full(len(train_df), -1, dtype=np.int32)
    next_burst = 0
    for sp in ["hyena", "leopard"]:
        sp_idx = np.where(train_df["species"].values == sp)[0]
        ids_sp = train_df["identity"].values[sp_idx]
        for ident in np.unique(ids_sp):
            sub_idx = sp_idx[ids_sp == ident]
            if len(sub_idx) == 1:
                burst_id[sub_idx[0]] = next_burst
                next_burst += 1
            else:
                S = X_norm[sub_idx] @ X_norm[sub_idx].T
                adj = (S >= burst_thresh)
                n_c, labels = connected_components(adj, directed=False)
                for c in range(n_c):
                    burst_id[sub_idx[labels == c]] = next_burst
                    next_burst += 1

    # Assign each identity to an `unseen_fold` in 0..n_splits-1, stratified by species & burst count
    unseen_fold_map = {}
    enc_fold_arr = np.full(len(train_df), -1, dtype=np.int32)

    for sp in ["hyena", "leopard"]:
        sp_df = train_df[train_df["species"] == sp].copy()
        sp_df["burst"] = burst_id[sp_df.index]
        id_stats = sp_df.groupby("identity").agg(
            n_rows=("id", "count"),
            n_bursts=("burst", "nunique")
        ).reset_index()
        # Sort by n_bursts, n_rows + tiny jitter so folds are balanced in both IDs and rows
        id_stats["jitter"] = rng.uniform(0, 1, size=len(id_stats))
        id_stats = id_stats.sort_values(["n_bursts", "n_rows", "jitter"], ascending=[False, False, True]).reset_index(drop=True)

        # Snake assignment across folds 0..n_splits-1 for tight balance
        for i, row in id_stats.iterrows():
            block = i // n_splits
            pos = i % n_splits
            f_idx = pos if (block % 2 == 0) else (n_splits - 1 - pos)
            unseen_fold_map[row["identity"]] = int(f_idx)

        # For identities with >= 2 bursts, assign their bursts across folds 0..n_splits-1
        # ensuring that in every fold k != unseen_fold[ident], at least 1 burst stays in gallery!
        for ident, grp in sp_df.groupby("identity"):
            bursts = grp.groupby("burst").size().sort_values(ascending=False)
            b_list = list(bursts.index)
            if len(b_list) < 2:
                continue  # Single-burst IDs stay in gallery when not held out as unseen
            u_fold = unseen_fold_map[ident]
            other_folds = [f for f in range(n_splits) if f != u_fold]
            rng.shuffle(other_folds)
            # Assign bursts round-robin across `other_folds` (and u_fold if >4 bursts)
            # Crucially: since len(b_list) >= 2 and we cycle through >=2 distinct folds,
            # no single fold k != u_fold can ever take ALL bursts of `ident`!
            cycle_folds = other_folds if len(b_list) <= len(other_folds) else (other_folds + [u_fold])
            for j, b_id in enumerate(b_list):
                assigned_f = cycle_folds[j % len(cycle_folds)]
                enc_fold_arr[grp.index[grp["burst"] == b_id]] = int(assigned_f)

    folds = []
    for k in range(n_splits):
        gal_idx = []
        val_idx = []
        val_labels = []
        val_is_new = []
        for idx in range(len(train_df)):
            ident = train_df["identity"].iat[idx]
            if unseen_fold_map[ident] == k:
                val_idx.append(idx)
                val_labels.append("new_individual")
                val_is_new.append(True)
            elif enc_fold_arr[idx] == k:
                val_idx.append(idx)
                val_labels.append(ident)
                val_is_new.append(False)
            else:
                gal_idx.append(idx)

        # Sanity-check: every known label in val_labels MUST exist in gal_idx!
        gal_ids = set(train_df["identity"].values[gal_idx])
        for lbl, is_n in zip(val_labels, val_is_new):
            if not is_n:
                assert lbl in gal_ids, f"Leak/Bug: known val label {lbl} missing from gallery in fold {k}!"
            else:
                assert lbl == "new_individual"

        folds.append({
            "fold": k,
            "gallery_idx": gal_idx,
            "val_idx": val_idx,
            "val_labels": val_labels,
            "val_is_new": val_is_new,
        })

    save_data = {"n_splits": n_splits, "burst_thresh": burst_thresh, "seed": seed, "folds": folds}
    with open(cache_path, "w") as f:
        json.dump(save_data, f)

    for fold in save_data["folds"]:
        fold["gallery_idx"] = np.array(fold["gallery_idx"], dtype=np.int32)
        fold["val_idx"] = np.array(fold["val_idx"], dtype=np.int32)
        fold["val_is_new"] = np.array(fold["val_is_new"], dtype=bool)
    return save_data


def evaluate_cv_predictions(cv_data, train_df, fold_preds_dict, verbose=True):
    """
    fold_preds_dict: dict mapping fold_idx -> list of top-5 prediction lists aligned with fold['val_idx'].
    Returns rich dictionary of exact metrics including species-level known/new MAP@5 and Test-LB Proxy MAP@5.
    """
    records = []
    for fold in cv_data["folds"]:
        k = fold["fold"]
        preds = fold_preds_dict[k]
        v_idx = fold["val_idx"]
        v_lbls = fold["val_labels"]
        v_new = fold["val_is_new"]
        sps = train_df["species"].values[v_idx]
        for i in range(len(v_idx)):
            records.append({
                "fold": k,
                "row_idx": int(v_idx[i]),
                "species": sps[i],
                "is_new": bool(v_new[i]),
                "ap5": ap_at_5(v_lbls[i], preds[i]),
            })
    df_res = pd.DataFrame(records)
    out = {"raw_oof_map5": float(df_res["ap5"].mean())}

    lb_proxy = 0.0
    for sp in ["hyena", "leopard"]:
        sub = df_res[df_res["species"] == sp]
        m_known = float(sub[~sub["is_new"]]["ap5"].mean())
        m_new = float(sub[sub["is_new"]]["ap5"].mean())
        pi = EST_PI_NEW[sp]
        m_cal = (1.0 - pi) * m_known + pi * m_new
        out[f"{sp}_raw_map5"] = float(sub["ap5"].mean())
        out[f"{sp}_known_map5"] = m_known
        out[f"{sp}_new_map5"] = m_new
        out[f"{sp}_proxy_map5"] = m_cal
        lb_proxy += TEST_SPECIES_WEIGHTS[sp] * m_cal

    out["lb_proxy_map5"] = float(lb_proxy)
    # Also compute sensitivity at lower/higher pi_new (-0.08 / +0.08)
    lb_low, lb_high = 0.0, 0.0
    for sp in ["hyena", "leopard"]:
        mk, mn = out[f"{sp}_known_map5"], out[f"{sp}_new_map5"]
        pi_lo = max(0.05, EST_PI_NEW[sp] - 0.08)
        pi_hi = min(0.90, EST_PI_NEW[sp] + 0.08)
        lb_low += TEST_SPECIES_WEIGHTS[sp] * ((1 - pi_lo) * mk + pi_lo * mn)
        lb_high += TEST_SPECIES_WEIGHTS[sp] * ((1 - pi_hi) * mk + pi_hi * mn)
    out["lb_proxy_low_pi"] = float(lb_low)
    out["lb_proxy_high_pi"] = float(lb_high)

    if verbose:
        print(
            f"CV LB-Proxy MAP@5: {out['lb_proxy_map5']:.5f} "
            f"[pi-8%: {out['lb_proxy_low_pi']:.5f}, pi+8%: {out['lb_proxy_high_pi']:.5f}] | "
            f"Hyena: {out['hyena_proxy_map5']:.5f} (known={out['hyena_known_map5']:.4f}, new={out['hyena_new_map5']:.4f}) | "
            f"Leopard: {out['leopard_proxy_map5']:.5f} (known={out['leopard_known_map5']:.4f}, new={out['leopard_new_map5']:.4f})"
        )
    return out


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    cv = build_or_load_cv(force_rebuild=True)
    for fold in cv["folds"]:
        v_idx = fold["val_idx"]
        v_new = fold["val_is_new"]
        sps = train_df["species"].values[v_idx]
        h_mask = (sps == "hyena")
        l_mask = (sps == "leopard")
        print(
            f"Fold {fold['fold']}: gallery={len(fold['gallery_idx'])}, val={len(v_idx)} | "
            f"hyena val={h_mask.sum()} (new={v_new[h_mask].mean():.1%}), "
            f"leopard val={l_mask.sum()} (new={v_new[l_mask].mean():.1%})"
        )
