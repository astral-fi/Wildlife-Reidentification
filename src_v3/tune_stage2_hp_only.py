"""Run only v3's Optuna stage-2 hyperparameter search (optuna_stage2_and_cluster_v3.optuna_tune_stage2_v3,
same call and trial budget as that script's main) and write cache_v3/optuna_stage2_3_v3_best_params.json.
optimize_stage3_regularized_v3.py only reads its "stage2_hp" key and retrains everything itself, so the
stage-C meta search and 480-model inference in optuna_stage2_and_cluster_v3.main are skipped.
"""
import os, sys, json, pickle, time
sys.path.insert(0, "src_v3"); sys.path.insert(0, "src_v2")
from optuna_stage2_and_cluster_v3 import optuna_tune_stage2_v3
from cv_v3 import CACHE_V3_DIR

all_hp = {}
for sp in ["hyena", "leopard"]:
    t0 = time.time()
    data = pickle.load(open(os.path.join(CACHE_V3_DIR, f"v3_bundles_{sp}.pkl"), "rb"))
    all_hp[sp] = optuna_tune_stage2_v3(sp, data["full_seed_bundles"][2026], n_trials_tree=20)
    print(f"[{sp}] stage-2 HPO done in {time.time() - t0:.0f}s", flush=True)
    del data
with open(os.path.join(CACHE_V3_DIR, "optuna_stage2_3_v3_best_params.json"), "w") as f:
    json.dump({"stage2_hp": all_hp}, f, indent=2)
print("wrote", os.path.join(CACHE_V3_DIR, "optuna_stage2_3_v3_best_params.json"))
