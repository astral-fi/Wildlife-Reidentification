import os
import json
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize

DATA_DIR = "reid_student_data_main"
CACHE_V2_DIR = "cache_v2"

# Exact test.csv species shares & KDE-validated open-set priors (confirmed by LB 0.82895 vs 0.78087!)
TEST_SPECIES_WEIGHTS = {
    "hyena": 1217.0 / 2791.0,
    "leopard": 1574.0 / 2791.0,
}
PRIOR_PI_NEW = {
    "hyena": 0.405,
    "leopard": 0.595,
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


def map_at_5_vec(cand_names_top5, y_true, is_new, sample_weight=None):
    aps = np.array([ap_at_5(yt, yp) for yt, yp in zip(y_true, cand_names_top5)], dtype=np.float64)
    if sample_weight is None:
        return float(aps.mean())
    sw = np.asarray(sample_weight, dtype=np.float64)
    return float(np.sum(aps * sw) / np.sum(sw))


def compute_encounter_ids(X_norm, identities, enc_thresh=0.80, small_thresh=0.94):
    """
    Singleton-preserving encounter clustering:
      - Large classes (n >= 4): group same-encounter bursts at `enc_thresh=0.80` (eliminates burst leak!)
      - Small classes (2 <= n <= 3): only group near-duplicates (`>= 0.94`) so `n=1` gallery classes
        genuinely appear as positive `known` queries during CV and sub-fold training!
    """
    enc_id = np.full(len(identities), -1, dtype=np.int32)
    nxt = 0
    for ident in np.unique(identities):
        idx = np.where(identities == ident)[0]
        if len(idx) == 1:
            enc_id[idx[0]] = nxt
            nxt += 1
        else:
            th = small_thresh if len(idx) <= 3 else enc_thresh
            adj = (X_norm[idx] @ X_norm[idx].T) >= th
            nc, lbl = connected_components(adj, directed=False)
            for c in range(nc):
                enc_id[idx[lbl == c]] = nxt
                nxt += 1
    return enc_id


def build_open_set_splits(identities, species_arr, X_norm, n_splits=5, enc_thresh=0.80, seed=42):
    """
    Leak-free Encounter-Grouped + Singleton-Preserving Open-Set Splits.
    Works identically for outer 5-fold CV AND nested internal sub-folds!
    """
    rng = np.random.RandomState(seed)
    N = len(identities)
    enc_id = np.full(N, -1, dtype=np.int32)
    nxt_off = 0
    for sp in np.unique(species_arr):
        m = np.where(species_arr == sp)[0]
        e_sp = compute_encounter_ids(X_norm[m], identities[m], enc_thresh=enc_thresh)
        enc_id[m] = e_sp + nxt_off
        nxt_off = int(enc_id[m].max()) + 1

    unseen_fold = {}
    enc_fold = np.full(N, -1, dtype=np.int32)

    df_tmp = pd.DataFrame({
        "pos": np.arange(N),
        "species": species_arr,
        "identity": identities,
        "enc": enc_id,
    })

    for sp, sp_df in df_tmp.groupby("species"):
        id_stats = sp_df.groupby("identity").agg(n_rows=("pos", "count"), n_enc=("enc", "nunique")).reset_index()
        id_stats["jit"] = rng.uniform(0, 1, size=len(id_stats))
        id_stats = id_stats.sort_values(["n_enc", "n_rows", "jit"], ascending=[False, False, True]).reset_index(drop=True)
        for i, r in id_stats.iterrows():
            blk, pos = i // n_splits, i % n_splits
            unseen_fold[r["identity"]] = int(pos if blk % 2 == 0 else (n_splits - 1 - pos))

        for ident, grp in sp_df.groupby("identity"):
            e_sizes = grp.groupby("enc").size().sort_values(ascending=False)
            e_list = list(e_sizes.index)
            if len(e_list) < 2:
                continue
            uf = unseen_fold[ident]
            others = [f for f in range(n_splits) if f != uf]
            rng.shuffle(others)
            cyc = others if len(e_list) <= len(others) else (others + [uf])
            for j, eid in enumerate(e_list):
                enc_fold[grp["pos"].values[grp["enc"].values == eid]] = int(cyc[j % len(cyc)])

    folds = []
    for k in range(n_splits):
        g_idx, v_idx, v_lbl, v_new = [], [], [], []
        for i in range(N):
            ident = identities[i]
            if unseen_fold[ident] == k:
                v_idx.append(i)
                v_lbl.append("new_individual")
                v_new.append(True)
            elif enc_fold[i] == k:
                v_idx.append(i)
                v_lbl.append(ident)
                v_new.append(False)
            else:
                g_idx.append(i)

        gal_set = set(identities[g_idx])
        for lbl, isn in zip(v_lbl, v_new):
            if not isn:
                assert lbl in gal_set, f"Known label {lbl} missing in gallery fold {k}!"
        folds.append({
            "fold": k,
            "gallery_idx": np.array(g_idx, dtype=np.int32),
            "val_idx": np.array(v_idx, dtype=np.int32),
            "val_labels": np.array(v_lbl, dtype=object),
            "val_is_new": np.array(v_new, dtype=bool),
        })
    return folds


def build_or_load_cv_v2(n_splits=5, enc_thresh=0.80, seed=42, force_rebuild=False):
    os.makedirs(CACHE_V2_DIR, exist_ok=True)
    path = os.path.join(CACHE_V2_DIR, f"cv_v2_k{n_splits}_e{int(enc_thresh*100)}_s{seed}.json")
    if os.path.exists(path) and not force_rebuild:
        with open(path, "r") as f:
            raw = json.load(f)
        for fd in raw:
            fd["gallery_idx"] = np.array(fd["gallery_idx"], dtype=np.int32)
            fd["val_idx"] = np.array(fd["val_idx"], dtype=np.int32)
            fd["val_labels"] = np.array(fd["val_labels"], dtype=object)
            fd["val_is_new"] = np.array(fd["val_is_new"], dtype=bool)
        return raw

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    miew_tr = np.load(os.path.join(DATA_DIR, "miewid_features.npy"))[:len(train_df)]
    X_norm = normalize(miew_tr, axis=1)
    folds = build_open_set_splits(
        train_df["identity"].values, train_df["species"].values, X_norm,
        n_splits=n_splits, enc_thresh=enc_thresh, seed=seed
    )
    serial = []
    for fd in folds:
        serial.append({
            "fold": fd["fold"],
            "gallery_idx": fd["gallery_idx"].tolist(),
            "val_idx": fd["val_idx"].tolist(),
            "val_labels": fd["val_labels"].tolist(),
            "val_is_new": fd["val_is_new"].tolist(),
        })
    with open(path, "w") as f:
        json.dump(serial, f)
    return folds


def eval_cv_v2(folds, train_df, fold_preds_dict, verbose=True, tag=""):
    recs = []
    for fd in folds:
        k = fd["fold"]
        vi = fd["val_idx"]
        vl = fd["val_labels"]
        vn = fd["val_is_new"]
        sps = train_df["species"].values[vi]
        pr = fold_preds_dict[k]
        for i in range(len(vi)):
            recs.append({"sp": sps[i], "is_new": bool(vn[i]), "ap": ap_at_5(vl[i], pr[i])})
    df = pd.DataFrame(recs)
    out = {}
    tot = 0.0
    for sp in ["hyena", "leopard"]:
        sub = df[df["sp"] == sp]
        mk = float(sub[~sub["is_new"]]["ap"].mean())
        mn = float(sub[sub["is_new"]]["ap"].mean())
        pi = PRIOR_PI_NEW[sp]
        mp = (1.0 - pi) * mk + pi * mn
        out[f"{sp}_known"] = mk
        out[f"{sp}_new"] = mn
        out[f"{sp}_map5"] = mp
        tot += TEST_SPECIES_WEIGHTS[sp] * mp
    out["lb_proxy"] = float(tot)
    if verbose:
        print(
            f"[{tag:32s}] V2 CV LB-Proxy: {out['lb_proxy']:.5f} | "
            f"Hyena: {out['hyena_map5']:.5f} (known={out['hyena_known']:.4f}, new={out['hyena_new']:.4f}) | "
            f"Leopard: {out['leopard_map5']:.5f} (known={out['leopard_known']:.4f}, new={out['leopard_new']:.4f})"
        )
    return out


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    folds = build_or_load_cv_v2(force_rebuild=True)
    for fd in folds:
        vi = fd["val_idx"]
        vn = fd["val_is_new"]
        sps = train_df["species"].values[vi]
        hm, lm = (sps == "hyena"), (sps == "leopard")
        # Also count how many known val queries belong to gallery classes of size == 1!
        g_vc = pd.Series(train_df["identity"].values[fd["gallery_idx"]]).value_counts()
        k_lbls = fd["val_labels"][~vn]
        n_sing_hits = sum(1 for lbl in k_lbls if g_vc.get(lbl, 0) == 1)
        print(
            f"Fold {fd['fold']}: gal={len(fd['gallery_idx'])}, val={len(vi)} | "
            f"hyena val={hm.sum()} (new={vn[hm].mean():.1%}), leopard val={lm.sum()} (new={vn[lm].mean():.1%}) | "
            f"n=1 gallery-singleton known queries={n_sing_hits}"
        )
