"""
V3 INFERENCE step (Kaggle "inference notebook"): load saved artifacts, predict, write submission.csv. No training.

Needs:
  - the artifacts written by `train_and_save_v3.py` (--artifacts, default artifacts_v3/)
  - a test bundle per species, from one of two sources (--test-bundle):
      rebuild (default): build it now with `features_v3.build_v3_feature_bundle` (the exact function training used)
                         from test.csv + the per-image features (MiewID, VLAD, Fisher, SIFT keypoint counts,
                         SIFT pair CSR) loaded by `smoke_test_v3.load_v3_raw_data`.
      cached:            read the prebuilt "test_bundle" from cache_v3/v3_bundles_{species}{suffix}.pkl.

Usage (from the repo root):
  python src_v3/infer_v3.py [--artifacts artifacts_v3] [--test-bundle rebuild|cached] [--out submission.csv]
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

from cv_v3 import DATA_DIR, CACHE_V3_DIR
from reproduce_v3_no_optuna import TOP4_OPEN, row_zscore, decode_top5, apply_symmetric_cluster_consensus
from stage2_v3_models import load_json, predict_stage2_run


def cached_test_bundles(species_list, suffixes):
    out = {}
    for sp in species_list:
        p = os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}{suffixes[sp]}.pkl")
        with open(p, "rb") as f:
            out[sp] = pickle.load(f)["test_bundle"]
    return out


def rebuild_test_bundles(species_list, stage1_cfg):
    from smoke_test_v3 import load_v3_raw_data
    from features_v3 import build_v3_feature_bundle

    train_df, test_df, all_id, all_vp, miew_all, miew_norms, _, vlad_all, fv_all, kp_counts, sift_csr = load_v3_raw_data()
    n_tr, n_te = len(train_df), len(test_df)
    test_glob = np.arange(n_tr, n_tr + n_te, dtype=np.int32)
    out = {}
    for sp in species_list:
        ts = time.time()
        full_g = np.where(train_df["species"].values == sp)[0].astype(np.int32)
        tm = test_df["species"].values == sp
        b = build_v3_feature_bundle(
            full_g, test_glob[tm], sp,
            all_id, all_vp, miew_all, miew_norms, vlad_all, fv_all, kp_counts,
            sift_csr, top_k_cands=20, stage1_cfg=stage1_cfg[sp],
        )
        b["te_pos"] = np.where(tm)[0]
        out[sp] = b
        print(f"[INFER V3] {sp}: test bundle rebuilt ({len(b['te_pos'])} queries) in {time.time() - ts:.1f}s", flush=True)
    return out


def predict_species(art, params, test_b, sp):
    s3 = params["stage3"][sp]
    p, cal_p, ow = s3["stage3"], s3["platt"], s3["open_weights"]
    runs = art["runs"]
    ws = 1.0 / len(runs)
    cand_ens, open_ens = None, None
    for run in runs:
        c_sd, o_sd = predict_stage2_run(run, test_b, sp)
        if cand_ens is None:
            cand_ens = {k: v * ws for k, v in c_sd.items()}
            open_ens = {k: v * ws for k, v in o_sd.items()}
        else:
            for k in c_sd:
                cand_ens[k] += c_sd[k] * ws
            for k in o_sd:
                open_ens[k] += o_sd[k] * ws

    c_rnk = 0.44 * cand_ens["cat_rank"] + 0.28 * cand_ens["lgb_rank"] + 0.28 * cand_ens["xgb_rank"]
    c_bin = (
        0.27 * row_zscore(cand_ens["hgb_bin"]) + 0.25 * row_zscore(cand_ens["cat_bin"])
        + 0.24 * row_zscore(cand_ens["lgb_bin"]) + 0.19 * row_zscore(cand_ens["xgb_bin"])
        + 0.05 * row_zscore(cand_ens["et_bin"])
    )
    c_tot = (1.0 - p["mix_rank"]) * row_zscore(c_bin) + p["mix_rank"] * row_zscore(c_rnk)
    o_tot = np.zeros(len(c_tot), dtype=np.float64)
    for ok in TOP4_OPEN:
        a, b = cal_p[ok]
        o_tot += ow[ok] * (a * open_ens[ok] + b)

    cn, cp, ol = apply_symmetric_cluster_consensus(test_b, c_tot, o_tot, p)
    return decode_top5(cn, cp, ol, delta=p["delta"], raw_s1=test_b["raw_s1"], max_mov_inl=test_b["max_mov_inl"], z60=test_b["z60"])


def check_submission(sub, test_df):
    assert list(sub["id"]) == list(test_df["id"]), "ids / order differ from test.csv"
    toks = sub["predictions"].str.split(" ")
    assert toks.map(len).eq(5).all(), "every row needs exactly 5 predictions"
    assert toks.map(lambda t: len(set(t)) == 5).all(), "duplicate predictions within a row"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_v3")
    ap.add_argument("--test-bundle", choices=["rebuild", "cached"], default="rebuild")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    t0 = time.time()

    params = load_json(os.path.join(args.artifacts, "v3_params.json"))
    species_list = ["hyena", "leopard"]
    arts = {sp: joblib.load(os.path.join(args.artifacts, f"v3_stage2_{sp}.joblib")) for sp in species_list}
    print(f"[INFER V3] artifacts loaded in {time.time() - t0:.1f}s", flush=True)

    suffixes = {sp: arts[sp]["bundle_suffix"] for sp in species_list}
    if args.test_bundle == "rebuild":
        if any(suffixes.values()):
            raise SystemExit(f"Models were trained on bundles with suffix {suffixes}; the rebuild path only builds the "
                             "plain features_v3 bundle. Retrain without --bundle-suffix or use --test-bundle cached.")
        test_bundles = rebuild_test_bundles(species_list, params["stage1_cfg"])
    else:
        test_bundles = cached_test_bundles(species_list, suffixes)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_preds = [None] * len(test_df)
    for sp in species_list:
        ts = time.time()
        test_b = test_bundles[sp]
        for loc_i, pred in enumerate(predict_species(arts[sp], params, test_b, sp)):
            test_preds[test_b["te_pos"][loc_i]] = " ".join(pred)
        print(f"[INFER V3] {sp}: predicted {len(test_b['te_pos'])} queries in {time.time() - ts:.1f}s", flush=True)

    sub = pd.DataFrame({"id": test_df["id"].values, "predictions": test_preds})
    check_submission(sub, test_df)
    sub.to_csv(args.out, index=False)
    print(f"[INFER V3] wrote {args.out} in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
