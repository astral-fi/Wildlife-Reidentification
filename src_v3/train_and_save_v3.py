"""
V3 TRAINING step (Kaggle "training notebook"): fit every Stage-2 model once and save everything inference needs.

Reads only training-side data:
  - cache_v3/v3_bundles_{species}{suffix}.pkl -> "full_seed_bundles" (open-set subfolds over the full train set)
  - the three V3 param JSONs (stage-1 feature cfg, stage-2 hyperparameters, stage-3 / Platt / open-weights)

Writes to --out (default: artifacts_v3/; on Kaggle pass --out /kaggle/working/artifacts_v3):
  - v3_stage2_hyena.joblib, v3_stage2_leopard.joblib : 3 runs x 5 subfolds x (8 candidate + 4 open-set) models
  - v3_params.json : every parameter inference needs, so it never has to read the cache JSONs

Usage (from the repo root):
  python src_v3/train_and_save_v3.py [--out artifacts_v3] [--bundle-suffix _exh]
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib

from cv_v3 import PRIOR_PI_NEW_V3, CACHE_V3_DIR
from stage2_v3_models import RUN_SPECS, PARAM_FILES, find_param_file, load_json, fit_stage2_run


def bundle_path(sp, suffix):
    """The exh suffix only exists for leopard; other species fall back to the plain bundle (same as reproduce_v3)."""
    p = os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}{suffix}.pkl")
    if suffix and os.path.exists(p):
        return p, suffix
    return os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}.pkl"), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts_v3")
    ap.add_argument("--bundle-suffix", default=os.environ.get("V3_BUNDLE_SUFFIX", ""),
                    help="'' (default) or '_exh' for the exhaustive-SIFT leopard bundles")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    hp_all = load_json(find_param_file(PARAM_FILES["stage2"]))["stage2_hp"]
    s3_all = load_json(find_param_file(PARAM_FILES["stage3"]))
    s1_all = load_json(find_param_file(PARAM_FILES["stage1"]))

    for sp in ["hyena", "leopard"]:
        ts = time.time()
        path, suffix = bundle_path(sp, args.bundle_suffix)
        with open(path, "rb") as f:
            full_seed_bundles = pickle.load(f)["full_seed_bundles"]
        runs = []
        for split_sd, model_sd in RUN_SPECS:
            tr = time.time()
            runs.append(fit_stage2_run(full_seed_bundles[split_sd], sp, hp_all[sp], seed=model_sd))
            print(f"[TRAIN V3] {sp}: run split={split_sd} model={model_sd} fitted in {time.time() - tr:.1f}s", flush=True)
        del full_seed_bundles

        out_p = os.path.join(args.out, f"v3_stage2_{sp}.joblib")
        joblib.dump({"species": sp, "bundle_suffix": suffix, "run_specs": RUN_SPECS, "runs": runs}, out_p, compress=3)
        print(f"[TRAIN V3] {sp}: saved {out_p} ({os.path.getsize(out_p) / 1e6:.1f} MB) in {time.time() - ts:.1f}s", flush=True)
        del runs

    params = {
        "stage1_cfg": s1_all,
        "stage2_hp": hp_all,
        "stage3": s3_all,
        "prior_pi_new": PRIOR_PI_NEW_V3,
        "run_specs": RUN_SPECS,
        "bundle_suffix": args.bundle_suffix,
    }
    with open(os.path.join(args.out, "v3_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"[TRAIN V3] All artifacts written to {args.out}/ in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
