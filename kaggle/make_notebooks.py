"""Generates the two Kaggle notebooks (training + inference) for the V3 pipeline. Run from the repo root."""
import nbformat as nbf

SETUP = r'''
import glob, os, shutil, sys, time, runpy, platform

INPUT = os.environ.get("WILDREID_INPUT", "/kaggle/input")      # Kaggle mounts datasets / notebook outputs here
WORK = os.environ.get("WILDREID_WORK", "/kaggle/working")      # saved as this notebook's output
RUN = os.environ.get("WILDREID_RUN", "/tmp/wildreid_run")      # scratch layout the scripts run in (not saved)
SPECIES = os.environ.get("WILDREID_SPECIES", "hyena leopard").split()
VERSION = "__VERSION__"
ART_NAME = f"artifacts_{VERSION}"

def find_one(pattern, what):
    hits = sorted(glob.glob(os.path.join(INPUT, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"Could not find {what} ({pattern}) under {INPUT}. Is it attached via 'Add Input'?")
    return os.path.dirname(hits[0])

def find_code_dataset():
    # V3 and V4 datasets share the code layout; V4 is the one with the exhaustive-SIFT leopard bundle.
    hits = sorted(glob.glob(os.path.join(INPUT, "**", "src_v3", "stage2_v3_models.py"), recursive=True))
    roots = [os.path.dirname(os.path.dirname(h)) for h in hits]
    roots = [r for r in roots if os.path.exists(os.path.join(r, "cache_v3", "v3_bundles_leopard_exh.pkl")) == (VERSION == "v4")]
    if not roots:
        raise FileNotFoundError(f"Could not find the wildreid-{VERSION} code dataset under {INPUT}. Is it attached via 'Add Input'?")
    return roots[0]

CODE_SRC = find_code_dataset()
DATA_SRC = find_one("test.csv", "the competition data")
print("code+cache dataset :", CODE_SRC)
print("competition data   :", DATA_SRC)

# The scripts use repo-relative paths (src_v2/, cache_v3/, reid_student_data_main/), so recreate that layout.
shutil.rmtree(RUN, ignore_errors=True)
os.makedirs(os.path.join(RUN, "cache_v3"))
for d in ("src_v2", "src_v3"):
    shutil.copytree(os.path.join(CODE_SRC, d), os.path.join(RUN, d))
for f in os.listdir(os.path.join(CODE_SRC, "cache_v3")):
    os.symlink(os.path.join(CODE_SRC, "cache_v3", f), os.path.join(RUN, "cache_v3", f))
os.symlink(DATA_SRC, os.path.join(RUN, "reid_student_data_main"))
os.chdir(RUN)
sys.path[:0] = [os.path.join(RUN, "src_v3"), os.path.join(RUN, "src_v2")]

# The cached feature bundles were pickled under NumPy 2 (module path numpy._core). Let NumPy 1.x read them too.
import numpy as np
if int(np.__version__.split(".")[0]) < 2:
    import numpy.core, numpy.core.multiarray
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)

import sklearn, lightgbm, xgboost, catboost, scipy, pandas
print("python", platform.python_version(), "| numpy", np.__version__, "| pandas", pandas.__version__,
      "| scipy", scipy.__version__, "| sklearn", sklearn.__version__, "| lightgbm", lightgbm.__version__,
      "| xgboost", xgboost.__version__, "| catboost", catboost.__version__, "| cpus", os.cpu_count())
'''

def nb(cells):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}}
    return n

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

def make(VERSION, TRAIN_EXTRA, TITLE_NOTE, DS):
  setup = SETUP.strip().replace("__VERSION__", VERSION)
  train = nb([
    md(f"""# Wildlife Re-ID {VERSION.upper()} — Training notebook

{TITLE_NOTE}

Fits every Stage-2 model **once** and saves everything inference needs to `/kaggle/working/artifacts_{VERSION}/`.
Run it with **Save Version → Save & Run All**; the saved files become this notebook's output, which the
inference notebook attaches as an input.

**Inputs (Add Input):**
- the competition data (`projectwild3`)
- the `{DS}` dataset: code (`src_v2/`, `src_v3/`) + precomputed V3 feature bundles and tuned parameters (`cache_v3/`)

**What gets trained** (per species): 3 runs (subfold-split seed × model seed) × 5 open-set subfolds ×
(3 listwise rankers + 5 binary rerankers + 4 open-set "new_individual" detectors) — LightGBM, XGBoost, CatBoost,
HistGradientBoosting and ExtraTrees with fixed, pre-tuned hyperparameters. Only competition-permitted libraries are used.

**Saved artifacts:** `v3_stage2_hyena.joblib`, `v3_stage2_leopard.joblib`, `v3_params.json`
(stage-1 feature config, stage-2 hyperparameters, stage-3 cluster-consensus / Platt / open-set weights, priors)."""),
    md("## 1. Locate inputs and set up the working layout"),
    code(setup),
    md("## 2. Train all models and save the artifacts\n\nCode: `src_v3/train_and_save_v3.py` → `src_v3/stage2_v3_models.py` (shared with the inference notebook)."),
    code(r'''ART = os.path.join(WORK, ART_NAME)
t0 = time.time()
sys.argv = ["train_and_save_v3.py", "--out", ART, "--species", *SPECIES] + ''' + repr(TRAIN_EXTRA) + r'''
runpy.run_path(os.path.join(RUN, "src_v3", "train_and_save_v3.py"), run_name="__main__")
print(f"training wall time: {(time.time() - t0) / 60:.1f} min")'''),
    md("## 3. Check the saved artifacts"),
    code(r'''import joblib, json
for f in sorted(os.listdir(ART)):
    print(f"{f:28s} {os.path.getsize(os.path.join(ART, f)) / 1e6:8.1f} MB")
for sp in SPECIES:
    a = joblib.load(os.path.join(ART, f"v3_stage2_{sp}.joblib"))
    n = sum(len(r["cand"]) * len(r["cand"][0]) + len(r["open"]) * len(r["open"][0]) for r in a["runs"])
    print(f"{sp}: {len(a['runs'])} runs, {n} fitted models")'''),
])

  infer = nb([
    md(f"""# Wildlife Re-ID {VERSION.upper()} — Inference notebook

{TITLE_NOTE}

Loads the artifacts saved by the training notebook and writes `/kaggle/working/submission.csv`. **No training happens here.**

**Inputs (Add Input):**
- the competition data (`projectwild3`)
- the `{DS}` dataset (same code + precomputed V3 feature bundles as training)
- the **output of the training notebook** (contains `artifacts_{VERSION}/`)

Test features are the test bundles built by `features_v3.build_v3_feature_bundle` — the same function and
stage-1 config that built the training bundles — from the provided MiewID descriptors and SIFT keypoints."""),
    md("## 1. Locate inputs and set up the working layout"),
    code(setup + "\n\nART = find_one(os.path.join(ART_NAME, \"v3_params.json\"), \"the training notebook's \" + ART_NAME + \" output\")\nprint(\"artifacts          :\", ART)"),
    md("## 2. Predict (timed)\n\nCode: `src_v3/infer_v3.py` → `src_v3/stage2_v3_models.py` (shared with the training notebook)."),
    code(r'''OUT = os.path.join(WORK, "submission.csv")
t0 = time.time()
sys.argv = ["infer_v3.py", "--artifacts", ART, "--test-bundle", "cached", "--out", OUT, "--species", *SPECIES]
runpy.run_path(os.path.join(RUN, "src_v3", "infer_v3.py"), run_name="__main__")
print(f"inference wall time: {time.time() - t0:.1f} s")'''),
    md("## 3. Check the submission file"),
    code(r'''import pandas as pd
sub = pd.read_csv(OUT)
test = pd.read_csv(os.path.join(DATA_SRC, "test.csv"))
print(sub.shape, "rows expected:", len(test))
print("all rows predicted:", sub["predictions"].notna().all())
print("new_individual ranked first: %.1f%%" % (100 * sub["predictions"].str.split().str[0].eq("new_individual").mean()))
sub.head(10)'''),
])

  nbf.write(train, f"kaggle/wildreid_{VERSION}_train.ipynb")
  nbf.write(infer, f"kaggle/wildreid_{VERSION}_infer.ipynb")
  print(f"wrote kaggle/wildreid_{VERSION}_train.ipynb and kaggle/wildreid_{VERSION}_infer.ipynb")


make("v3", [], "V3: MiewID + SIFT/VLAD/Fisher candidate features, stacked rerankers, open-set detectors, cluster consensus.", "wildreid-v3")
make("v4", ["--bundle-suffix", "_exh", "--stage3-file", "v3_final_polished_stage3_params_exhsift.json"],
     "V4 = V3 + exhaustive SIFT verification of each leopard query against up to 24 gallery images of each of its "
     "top-20 candidate identities (+7 candidate and +3 open-set leopard features), with re-polished stage-3 parameters. "
     "Hyena features are unchanged.", "wildreid-v4")
